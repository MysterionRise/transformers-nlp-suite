#!/usr/bin/env python3
"""
UI Launcher - Launch all NLP demo UIs and API server

This script provides an easy way to launch all available interactive UIs
and the production REST API server.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from ui.catalog import UIS as UI_CATALOG

PROJECT_ROOT = Path(__file__).resolve().parent

# Available UIs, derived from ui/catalog.py (kept as plain dicts for callers and tests)
UIS = {
    key: {
        "name": entry.title,
        "file": entry.file,
        "module": entry.import_path,
        "port": entry.port,
        "description": entry.description,
    }
    for key, entry in UI_CATALOG.items()
}

# All UIs in one Gradio app: one process, one port, one shared model cache
ALL_IN_ONE = {
    "name": "All demo UIs (combined app)",
    "file": "ui/app.py",
    "module": "ui.app",
    "port": 7860,
    "description": "Every UI as a tab in one app sharing a single model cache",
}

# API server configuration
API_CONFIG = {
    "name": "Transformers NLP Suite API",
    "host": "127.0.0.1",
    "port": 8000,
    "description": "API-first customer intelligence service with auth, rate limiting, metrics, and Swagger docs",
}


def print_banner():
    """Print welcome banner"""
    print("=" * 70)
    print("  Transformers NLP Suite - UI & API Launcher")
    print("=" * 70)
    print()


def list_uis():
    """List all available UIs and API server"""
    print("Available UIs:")
    print()
    for key, info in UIS.items():
        print(f"  {key:15} - {info['name']}")
        print(f"  {' ' * 15}   {info['description']}")
        print(f"  {' ' * 15}   Port: {info['port']}")
        print()

    print(f"  {'all':15} - {ALL_IN_ONE['name']}")
    print(f"  {' ' * 15}   {ALL_IN_ONE['description']}")
    print(f"  {' ' * 15}   Port: {ALL_IN_ONE['port']} (override with --port)")
    print()

    print("API Server:")
    print()
    print(f"  {'api':15} - {API_CONFIG['name']}")
    print(f"  {' ' * 15}   {API_CONFIG['description']}")
    print(f"  {' ' * 15}   Port: {API_CONFIG['port']}")
    print(f"  {' ' * 15}   Docs: http://{API_CONFIG['host']}:{API_CONFIG['port']}/docs")
    print()


def launch_ui(ui_key: str, host: Optional[str] = None, port: Optional[int] = None):
    """
    Launch one UI (or ``all`` for the combined app) as ``python -m ui.<module>``.

    Runs from the repository root so imports work from any working directory.
    ``host`` and ``port`` are forwarded as ``--host`` / ``--port``; without them the
    UI binds to NLP_SERVER_NAME and its catalog port (NLP_SERVER_PORT for ``all``).
    """
    if ui_key == "all":
        ui_info = ALL_IN_ONE
    elif ui_key in UIS:
        ui_info = UIS[ui_key]
    else:
        print(f"Error: Unknown UI '{ui_key}'")
        print()
        list_uis()
        return False

    ui_file = PROJECT_ROOT / ui_info["file"]
    if not ui_file.exists():
        print(f"Error: UI file not found: {ui_file}")
        return False

    shown_host = host or os.environ.get("NLP_SERVER_NAME", "127.0.0.1")
    shown_port = port or ui_info["port"]

    print(f"Launching {ui_info['name']}...")
    print(f"URL: http://{shown_host}:{shown_port}")
    print()
    print("Press Ctrl+C to stop the server")
    print("-" * 70)
    print()

    command = [sys.executable, "-m", ui_info["module"]]
    if host:
        command += ["--host", host]
    if port:
        command += ["--port", str(port)]
    try:
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
        return True
    except KeyboardInterrupt:
        print("\nShutting down...")
        return True
    except Exception as e:
        print(f"Error launching UI: {e}")
        return False


def launch_api(host: str = None, port: int = None, reload: bool = False):
    """Launch the FastAPI server"""
    host = host or API_CONFIG["host"]
    port = port or API_CONFIG["port"]

    print(f"Launching {API_CONFIG['name']}...")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"API URL: http://{host}:{port}")
    print(f"Swagger Docs: http://{host}:{port}/docs")
    print(f"ReDoc: http://{host}:{port}/redoc")
    print()
    print("Press Ctrl+C to stop the server")
    print("-" * 70)
    print()

    try:
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "api.main:app",
            "--host",
            host,
            "--port",
            str(port),
        ]
        if reload:
            cmd.append("--reload")

        subprocess.run(cmd, check=True)
        return True
    except KeyboardInterrupt:
        print("\nShutting down API server...")
        return True
    except Exception as e:
        print(f"Error launching API server: {e}")
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Launch the Transformers NLP Suite API server or its demo UIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch_ui.py sentiment              # Launch sentiment analysis UI
  python launch_ui.py qa                     # Launch question answering UI
  python launch_ui.py all                    # Every UI as tabs in one app (port 7860)
  python launch_ui.py all --host 0.0.0.0     # ...reachable from other machines/containers
  python launch_ui.py api                    # Launch REST API server
  python launch_ui.py api --reload           # Launch API with auto-reload
  python launch_ui.py --list                 # List all available options
  python launch_ui.py                        # Show menu to select

Available UIs:
  sentiment      - Sentiment Analysis
  similarity     - Semantic Similarity
  ner            - Named Entity Recognition
  summarization  - Summarization
  performance    - Evaluation Metrics
  qa             - Question Answering
  generation     - Text Generation
  zero_shot      - Zero-Shot Classification
  translation    - Translation
  vision         - Vision-Language
  all            - Every UI in one app with a shared model cache

API Server:
  api            - REST API (Swagger docs at /docs)
        """,
    )

    parser.add_argument(
        "ui",
        nargs="?",
        choices=list(UIS.keys()) + ["all", "api"],
        help="UI to launch, 'all' for every UI in one app, or 'api' for the REST API",
    )

    parser.add_argument("--list", "-l", action="store_true", help="List all available UIs and API")

    # API-specific arguments
    parser.add_argument("--host", type=str, help="Bind address for the API or UI (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, help="Port for the API (default 8000) or UI (default: catalog port)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for API server (development)")

    args = parser.parse_args()

    print_banner()

    if args.list:
        list_uis()
        return 0

    if args.ui:
        if args.ui == "api":
            success = launch_api(host=args.host, port=args.port, reload=args.reload)
        else:
            success = launch_ui(args.ui, host=args.host, port=args.port)
        return 0 if success else 1

    # Interactive menu
    list_uis()
    print("Enter the name of the UI or 'api' to launch (or 'q' to quit):")
    print()

    while True:
        try:
            choice = input("Select> ").strip().lower()

            if choice in ["q", "quit", "exit"]:
                print("Goodbye!")
                return 0

            if choice == "api":
                launch_api()
                break
            elif choice in UIS or choice == "all":
                launch_ui(choice)
                break
            else:
                print(f"Unknown option: {choice}")
                print("Available options:", ", ".join(list(UIS.keys()) + ["all", "api"]))

        except KeyboardInterrupt:
            print("\nGoodbye!")
            return 0
        except EOFError:
            print("\nGoodbye!")
            return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
