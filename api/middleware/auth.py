"""
Authentication middleware for API security

Provides:
- API Key authentication (X-API-Key header)
- JWT Bearer token authentication
- Role-based access control
"""

from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel


class User(BaseModel):
    """Authenticated user model"""

    api_key: str
    name: str
    role: str  # "admin", "user", "readonly"
    rate_limit: int  # requests per minute
    enabled: bool = True


class TokenData(BaseModel):
    """JWT token payload"""

    sub: str  # subject (api_key or user_id)
    name: str
    role: str
    exp: datetime


# Security schemes. The scheme names are what the OpenAPI document references, so
# Swagger UI's "Authorize" dialog sends these credentials with "Try it out".
api_key_header = APIKeyHeader(
    name="X-API-Key",
    scheme_name="ApiKeyAuth",
    description="API key from NLP_API__API_KEYS",
    auto_error=False,
)
bearer_scheme = HTTPBearer(
    scheme_name="BearerAuth",
    bearerFormat="JWT",
    description="JWT from POST /api/v1/auth/token",
    auto_error=False,
)


def get_api_settings():
    """Get API settings - lazy import to avoid circular dependencies"""
    from config.settings import get_settings

    settings = get_settings()
    return getattr(settings, "api", None) or _get_default_api_settings()


def _get_default_api_settings():
    """Default API settings if not configured — no keys shipped by default"""

    class DefaultAPISettings:
        jwt_secret = ""
        jwt_algorithm = "HS256"
        jwt_expiration_hours = 24
        api_keys = {}

    return DefaultAPISettings()


def lookup_api_key(api_key: Optional[str]) -> Optional[User]:
    """
    Return the user for an enabled API key, or None. Records no metrics, so the
    rate limiter can call it without inflating auth_attempts_total.
    """
    if not api_key:
        return None
    user_data = get_api_settings().api_keys.get(api_key)
    if not isinstance(user_data, dict) or not user_data.get("enabled", True):
        return None
    return User(
        api_key=api_key,
        name=str(user_data.get("name", "unnamed")),
        role=str(user_data.get("role", "user")),
        rate_limit=int(user_data.get("rate_limit") or getattr(get_api_settings(), "default_rate_limit", 100)),
        enabled=True,
    )


def verify_api_key(api_key: str) -> Optional[User]:
    """
    Verify an API key and return the associated user

    Args:
        api_key: The API key to verify

    Returns:
        User object if valid, None otherwise
    """
    from utils.metrics import record_auth_attempt

    user = lookup_api_key(api_key)
    record_auth_attempt("api_key", user is not None)
    return user


def create_access_token(user: User, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token for a user

    Args:
        user: User object to create token for
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token string
    """
    api_settings = get_api_settings()
    if not api_settings.jwt_secret:
        raise RuntimeError("JWT signing is not configured: set NLP_API__JWT_SECRET")

    if expires_delta is None:
        expires_delta = timedelta(hours=api_settings.jwt_expiration_hours)

    expire = datetime.now(timezone.utc) + expires_delta

    to_encode = {
        "sub": user.api_key,
        "name": user.name,
        "role": user.role,
        "exp": expire,
    }

    encoded_jwt = jwt.encode(
        to_encode,
        api_settings.jwt_secret,
        algorithm=api_settings.jwt_algorithm,
    )

    return encoded_jwt


def jwt_subject(token: str) -> Optional[str]:
    """
    The API key a valid, unexpired JWT was issued for, or None. Records no metrics.

    Tokens must carry an ``exp`` claim, and nothing validates while
    NLP_API__JWT_SECRET is empty (fail closed).
    """
    api_settings = get_api_settings()
    if not api_settings.jwt_secret or not token:
        return None
    try:
        payload = jwt.decode(
            token,
            api_settings.jwt_secret,
            algorithms=[api_settings.jwt_algorithm],
            options={"require_exp": True},
        )
    except JWTError:
        return None
    subject = payload.get("sub")
    return subject if isinstance(subject, str) and subject else None


def verify_jwt_token(token: str) -> Optional[User]:
    """
    Verify a JWT token and return the associated user

    The underlying API key is re-checked on every request, so disabling a key
    also revokes the tokens issued for it.

    Args:
        token: JWT token string

    Returns:
        User object if valid, None otherwise
    """
    from utils.metrics import record_auth_attempt

    subject = jwt_subject(token)
    user = lookup_api_key(subject) if subject else None
    record_auth_attempt("jwt", user is not None)
    return user


async def get_current_user(
    request: Request,
    api_key: Annotated[Optional[str], Security(api_key_header)] = None,
    bearer_token: Annotated[Optional[HTTPAuthorizationCredentials], Security(bearer_scheme)] = None,
) -> User:
    """
    Dependency to get the current authenticated user

    Supports both API key (X-API-Key header) and JWT Bearer token authentication.
    API key authentication is checked first, then JWT.

    Args:
        request: FastAPI request object
        api_key: API key from X-API-Key header
        bearer_token: JWT token from Authorization header

    Returns:
        Authenticated User object

    Raises:
        HTTPException: If authentication fails
    """
    user = None

    # Try API key authentication first
    if api_key:
        user = verify_api_key(api_key)
        if user:
            # Store auth method for logging
            request.state.auth_method = "api_key"
            request.state.user = user
            return user

    # Try JWT Bearer token
    if bearer_token:
        user = verify_jwt_token(bearer_token.credentials)
        if user:
            request.state.auth_method = "jwt"
            request.state.user = user
            return user

    # No valid authentication provided
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "error": "unauthorized",
            "message": "Invalid or missing authentication credentials",
            "hint": "Provide either X-API-Key header or Authorization: Bearer <token>",
        },
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_optional_user(
    request: Request,
    api_key: Annotated[Optional[str], Security(api_key_header)] = None,
    bearer_token: Annotated[Optional[HTTPAuthorizationCredentials], Security(bearer_scheme)] = None,
) -> Optional[User]:
    """
    Dependency to get the current user if authenticated, or None

    Use this for endpoints that work with or without authentication,
    but may provide enhanced functionality when authenticated.

    Returns:
        User object if authenticated, None otherwise
    """
    try:
        return await get_current_user(request, api_key, bearer_token)
    except HTTPException:
        return None


def require_role(allowed_roles: list[str]):
    """
    Dependency factory to require specific roles

    Args:
        allowed_roles: List of role names that are allowed

    Returns:
        Dependency function that checks user role

    Example:
        @app.get("/admin")
        async def admin_endpoint(user: User = Depends(require_role(["admin"]))):
            ...
    """

    async def role_checker(user: Annotated[User, Depends(get_current_user)]) -> User:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "forbidden",
                    "message": f"Insufficient permissions. Required role: {allowed_roles}",
                    "your_role": user.role,
                },
            )
        return user

    return role_checker


class APIKeyAuth:
    """
    Callable class for API key authentication

    Can be used as a dependency directly:
        user: User = Depends(APIKeyAuth())
    """

    async def __call__(
        self,
        request: Request,
        api_key: Annotated[Optional[str], Security(api_key_header)] = None,
    ) -> User:
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": "unauthorized",
                    "message": "X-API-Key header is required",
                },
            )

        user = verify_api_key(api_key)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": "unauthorized",
                    "message": "Invalid API key",
                },
            )

        request.state.auth_method = "api_key"
        request.state.user = user
        return user


class JWTAuth:
    """
    Callable class for JWT-only authentication

    Can be used as a dependency directly:
        user: User = Depends(JWTAuth())
    """

    async def __call__(
        self,
        request: Request,
        bearer_token: Annotated[Optional[HTTPAuthorizationCredentials], Security(bearer_scheme)] = None,
    ) -> User:
        if not bearer_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": "unauthorized",
                    "message": "Authorization header with Bearer token is required",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        user = verify_jwt_token(bearer_token.credentials)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": "unauthorized",
                    "message": "Invalid or expired JWT token",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        request.state.auth_method = "jwt"
        request.state.user = user
        return user
