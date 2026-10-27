"""
Text Generation - continue a prompt with GPT-2 family models.

Models come from the ``text_generation`` registry category and load through the
shared model cache. Sampling is seeded with ``torch.manual_seed``, which is
process-global, so seeding and generating happen under one lock: two users with
the same seed and settings get the same text even when they generate at once.
"""

import sys
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gradio as gr
import pandas as pd
import torch

try:
    from ui import common as shell
except ModuleNotFoundError:  # running as `python ui/generation_playground.py`
    import common as shell  # type: ignore[no-redef]

logger = shell.get_logger(__name__)

UI_KEY = "generation"
MODELS = shell.ModelMenu("text_generation", default="gpt2")
COMPARE_DEFAULT = [MODELS.get("gpt2").name, MODELS.get("distilgpt2").name]

DEFAULT_NEW_TOKENS = 60
MAX_NEW_TOKENS = 200
MAX_SAMPLES = 3
MAX_BATCH_PROMPTS = 20
MAX_SEED = 2**32 - 1
DEFAULT_SEED = 42
BATCH_COLUMNS = ["#", "Prompt", "Continuation"]

# Guards torch.manual_seed + generation: the RNG is shared by every request in the process.
_GENERATE_LOCK = threading.Lock()


# --------------------------------------------------------------------------- validation


def _seed_value(seed: Any) -> Optional[int]:
    """None for an empty seed (random output), otherwise a validated integer seed."""
    if seed is None or (isinstance(seed, str) and not seed.strip()):
        return None
    try:
        value = int(float(seed))
    except (TypeError, ValueError, OverflowError):
        shell.fail(f"Seed must be a whole number between 0 and {MAX_SEED}.")
    if not 0 <= value <= MAX_SEED:
        shell.fail(f"Seed must be a whole number between 0 and {MAX_SEED}.")
    return value


def _in_range(value: Any, low: float, high: float, name: str, cast=float) -> Any:
    try:
        number = cast(value)
    except (TypeError, ValueError, OverflowError):
        shell.fail(f"{name} must be a number between {low} and {high}.")
    if not low <= number <= high:
        shell.fail(f"{name} must be between {low} and {high}.")
    return number


def _sampling(
    max_new_tokens: Any, temperature: Any, top_p: Any, top_k: Any = 50, num_return_sequences: Any = 1
) -> Dict[str, Any]:
    """Validated sampling settings shared by every tab."""
    return {
        "max_new_tokens": _in_range(max_new_tokens, 1, MAX_NEW_TOKENS, "New tokens", int),
        "temperature": _in_range(temperature, 0.01, 2.0, "Temperature"),
        "top_p": _in_range(top_p, 0.01, 1.0, "Top-p"),
        "top_k": _in_range(top_k, 0, 1000, "Top-k", int),
        "num_return_sequences": _in_range(num_return_sequences, 1, MAX_SAMPLES, "Samples", int),
    }


# --------------------------------------------------------------------------- inference


def _generate(generator: Any, prompt: str, seed: Optional[int], settings: Dict[str, Any]) -> List[str]:
    """Generate continuations of `prompt` (without the prompt itself)."""
    kwargs = dict(settings, do_sample=True, return_full_text=False)
    eos = getattr(generator.tokenizer, "eos_token_id", None)
    if eos is not None:  # GPT-2 style models have no pad token; pad with the model's own EOS
        kwargs["pad_token_id"] = eos
    with _GENERATE_LOCK:
        if seed is not None:
            torch.manual_seed(seed)
        outputs = generator(prompt, **kwargs)
    if outputs and isinstance(outputs[0], list):
        outputs = outputs[0]
    return [output["generated_text"] for output in outputs]


def _sample_html(prompt: str, continuation: str, title: str = "") -> str:
    heading = f'<span class="tns-label">{shell.esc(title)}</span><br>' if title else ""
    if not continuation.strip():
        continuation_html = '<span class="tns-label">the model stopped without adding text</span>'
    else:
        continuation_html = shell.esc(continuation)
    return (
        f'<div class="tns-text" style="margin-bottom: 0.75rem">{heading}'
        f"<strong>{shell.esc(prompt)}</strong>{continuation_html}</div>"
    )


# --------------------------------------------------------------------------- handlers


def generate_text(
    prompt: str,
    model_name: str,
    max_new_tokens: int = DEFAULT_NEW_TOKENS,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    num_return_sequences: int = 1,
    seed: Optional[int] = DEFAULT_SEED,
) -> Tuple[str, Dict[str, Any]]:
    """Continue one prompt; returns the samples as HTML (prompt in bold) and the run details."""
    prompt = shell.require_text(prompt, "a prompt")
    choice = MODELS.get(model_name)
    seed = _seed_value(seed)
    settings = _sampling(max_new_tokens, temperature, top_p, top_k, num_return_sequences)
    generator = MODELS.load(model_name)
    used = shell.clip_for_model(prompt, generator, label="Prompt", reserve=settings["max_new_tokens"])
    try:
        continuations = _generate(generator, used, seed, settings)
    except Exception as exc:
        shell.fail_from(exc, "Text generation")

    many = len(continuations) > 1
    html = "".join(
        _sample_html(used, text, f"Sample {index}" if many else "") for index, text in enumerate(continuations, 1)
    )
    details = {
        "model": choice.model_id,
        "seed": seed,
        "settings": settings,
        "prompt_truncated": used != prompt,
        "samples": [used + text for text in continuations],
    }
    return html, details


def compare_models(
    prompt: str,
    model_names: Sequence[str],
    max_new_tokens: int = DEFAULT_NEW_TOKENS,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: Optional[int] = DEFAULT_SEED,
) -> str:
    """Run the same prompt and seed through each selected model, one model at a time."""
    prompt = shell.require_text(prompt, "a prompt")
    names = list(dict.fromkeys(model_names or []))
    if not names:
        shell.fail("Select at least one model to compare.")
    choices = [MODELS.get(name) for name in names]
    seed = _seed_value(seed)
    settings = _sampling(max_new_tokens, temperature, top_p)
    cache_size = shell.get_settings().max_cached_models
    if len(choices) > cache_size:
        shell.warn(
            f"{len(choices)} models selected but the model cache holds {cache_size}. They run one after another "
            "and earlier models may be unloaded, so this is slower and the next run reloads them."
        )

    cards, failures = [], []
    for choice in choices:
        started = time.perf_counter()
        try:
            generator = MODELS.load(choice.name)
            used = shell.clip_for_model(prompt, generator, label="Prompt", reserve=settings["max_new_tokens"])
            [text] = _generate(generator, used, seed, settings)
        except Exception as exc:
            message = getattr(exc, "message", None) or str(exc)
            logger.warning(f"Comparison with {choice.name} failed: {message}")
            failures.append(f"{choice.name}: {message}")
            cards.append(
                f'<div class="tns-text" style="margin-bottom: 0.75rem"><span class="tns-label">'
                f"{shell.esc(choice.name)} · failed</span><br>{shell.esc(message[:300])}</div>"
            )
            continue
        cards.append(_sample_html(used, text, f"{choice.name} · {time.perf_counter() - started:.1f} s"))

    if len(failures) == len(choices):
        shell.fail("Every selected model failed. " + " ".join(failures)[:400])
    return "".join(cards)


def batch_generate(
    prompts: str,
    model_name: str,
    max_new_tokens: int = DEFAULT_NEW_TOKENS,
    temperature: float = 0.7,
    seed: Optional[int] = DEFAULT_SEED,
) -> Tuple[pd.DataFrame, str]:
    """Continue one prompt per line; a failing prompt is reported in its row."""
    items = shell.split_items(prompts, max_items=MAX_BATCH_PROMPTS, what="one prompt")
    choice = MODELS.get(model_name)
    seed = _seed_value(seed)
    settings = _sampling(max_new_tokens, temperature, 0.9)
    generator = MODELS.load(model_name)
    budget = max(1, shell.token_limit(generator) - settings["max_new_tokens"])

    rows, clipped, failed = [], 0, 0
    for index, item in enumerate(items, 1):
        used, _ = shell.truncate_to_tokens(item, generator.tokenizer, budget)
        clipped += used != item
        try:
            [text] = _generate(generator, used, seed, settings)
            rows.append([index, used, text])
        except Exception as exc:
            logger.warning(f"Batch prompt {index} failed: {exc}")
            failed += 1
            rows.append([index, used, f"error: {str(exc)[:120]}"])

    if clipped:
        shell.warn(f"{clipped} prompt(s) exceeded the model's {budget}-token prompt budget and were truncated.")
    summary = f"{len(items)} prompts continued with {choice.name}"
    summary += f" (seed {seed}, each prompt seeded the same way)." if seed is not None else " (random seed)."
    if failed:
        summary += f" **{failed} failed.**"
    return pd.DataFrame(rows, columns=BATCH_COLUMNS), summary


EXAMPLES = [
    ["The future of artificial intelligence", "GPT-2", 0.7],
    ["Once upon a time, in a galaxy far away", "GPT-2", 0.9],
    ["Scientists discovered a new species of", "DistilGPT-2", 0.6],
    ["The weather today is", "DistilGPT-2", 0.5],
]

ABOUT = f"""
The models continue your prompt one token at a time, sampling each token from the model's predicted distribution.

- **New tokens**: how much text to add (the prompt does not count). Prompts longer than the model's context minus
  the new tokens are cut, and you get a warning.
- **Temperature**: below 1 makes likely tokens more likely (focused, repetitive); above 1 flattens the distribution
  (varied, often incoherent).
- **Top-p / top-k**: sample only from the smallest set of tokens whose probabilities add up to p, or from the k most
  likely tokens (0 disables top-k).
- **Seed**: the same seed, model, prompt and settings give the same text. Clear it for a different result each run.

The comparison tab runs the models one after another with the same seed. The model cache holds
{shell.get_settings().max_cached_models} models; selecting more works but reloads models. GPT-2 models were trained
on web text from 2019: they can be wrong, repetitive or biased, and they do not follow instructions.
"""


def _settings_controls() -> Tuple[gr.Slider, gr.Slider, gr.Slider]:
    new_tokens = gr.Slider(10, MAX_NEW_TOKENS, value=DEFAULT_NEW_TOKENS, step=10, label="New tokens")
    temperature = gr.Slider(
        0.1, 2.0, value=0.7, step=0.1, label="Temperature", info="Lower = more focused, higher = more random"
    )
    top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p (nucleus)")
    return new_tokens, temperature, top_p


def _seed_input() -> gr.Number:
    return gr.Number(
        label="Seed",
        value=DEFAULT_SEED,
        precision=0,
        minimum=0,
        maximum=MAX_SEED,
        info="Same seed and settings give the same text; clear it for random output.",
    )


def create_ui() -> gr.Blocks:
    """Build the text generation page."""
    with shell.page(UI_KEY) as demo:
        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=3):
                    prompt_input = gr.Textbox(
                        label="Prompt", placeholder="Start a sentence and let the model continue it...", lines=5
                    )
                    model_dropdown = MODELS.dropdown()
                    generate_btn = gr.Button("Generate", variant="primary")
                with gr.Column(scale=2):
                    new_tokens, temperature, top_p = _settings_controls()
                    top_k = gr.Slider(0, 100, value=50, step=5, label="Top-k")
                    samples = gr.Slider(1, MAX_SAMPLES, value=1, step=1, label="Samples")
                    seed = _seed_input()
            output_html = gr.HTML(label="Generated text")
            with gr.Accordion("Details", open=False):
                output_json = gr.JSON(label="Run details")
            gr.Examples(examples=EXAMPLES, inputs=[prompt_input, model_dropdown, temperature], label="Try an example")

        with gr.Tab("Compare models"):
            gr.Markdown("Same prompt, settings and seed for each selected model, run one model at a time.")
            with gr.Row():
                with gr.Column(scale=3):
                    compare_prompt = gr.Textbox(label="Prompt", lines=5)
                    compare_models_input = gr.CheckboxGroup(
                        choices=MODELS.names, value=COMPARE_DEFAULT, label="Models to compare"
                    )
                    compare_btn = gr.Button("Compare models", variant="primary")
                with gr.Column(scale=2):
                    compare_tokens, compare_temperature, compare_top_p = _settings_controls()
                    compare_seed = _seed_input()
            compare_output = gr.HTML(label="Outputs")

        with gr.Tab("Batch"):
            gr.Markdown(f"One prompt per line, up to {MAX_BATCH_PROMPTS} prompts.")
            with gr.Row():
                with gr.Column(scale=2):
                    batch_prompts = gr.Textbox(
                        label="Prompts (one per line)", placeholder="Prompt 1\nPrompt 2", lines=8
                    )
                    batch_model = MODELS.dropdown()
                    batch_tokens = gr.Slider(10, MAX_NEW_TOKENS, value=DEFAULT_NEW_TOKENS, step=10, label="New tokens")
                    batch_temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
                    batch_seed = _seed_input()
                    batch_btn = gr.Button("Generate all", variant="primary")
                with gr.Column(scale=3):
                    batch_summary = gr.Markdown()
                    batch_output = gr.Dataframe(headers=BATCH_COLUMNS, label="Results", wrap=True, interactive=False)

        shell.about(ABOUT, MODELS)

        generate_btn.click(
            generate_text,
            inputs=[prompt_input, model_dropdown, new_tokens, temperature, top_p, top_k, samples, seed],
            outputs=[output_html, output_json],
        )
        compare_btn.click(
            compare_models,
            inputs=[
                compare_prompt,
                compare_models_input,
                compare_tokens,
                compare_temperature,
                compare_top_p,
                compare_seed,
            ],
            outputs=[compare_output],
        )
        batch_btn.click(
            batch_generate,
            inputs=[batch_prompts, batch_model, batch_tokens, batch_temperature, batch_seed],
            outputs=[batch_output, batch_summary],
        )
    return demo


if __name__ == "__main__":
    shell.launch(create_ui(), UI_KEY, argv=sys.argv[1:])
