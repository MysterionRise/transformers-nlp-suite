"""
Summarization - abstractive summaries from one model or several side by side.

Models come from the ``summarization`` registry category (BART, DistilBART, T5,
Pegasus) and load through the shared model cache. Each model's registry
``min_length``/``max_length`` are the length defaults, T5 models get their
``summarize: `` prefix, and input longer than the model's token limit is cut
with a warning instead of failing inside the model.
"""

import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import gradio as gr
import pandas as pd

try:
    from ui import common as shell
except ModuleNotFoundError:  # running as `python ui/summarization_studio.py`
    import common as shell  # type: ignore[no-redef]

logger = shell.get_logger(__name__)

UI_KEY = "summarization"
MODELS = shell.ModelMenu("summarization", default="bart_large_cnn")

MIN_WORDS = 30
FALLBACK_LENGTHS = (30, 130)
COMPARE_COLUMNS = ["Model", "Summary", "Words", "Seconds"]
COMPARE_DEFAULT = [MODELS.get("distilbart_cnn").name, MODELS.get("t5_base").name]

TAB_SUMMARIZE = "summarize"
TAB_COMPARE = "compare"
TAB_EXAMPLES = "examples"


def length_defaults(model_name: Optional[str]) -> Tuple[int, int]:
    """(min_length, max_length) in tokens from the model's registry entry."""
    config = MODELS.get(model_name).config
    return config.min_length or FALLBACK_LENGTHS[0], config.max_length or FALLBACK_LENGTHS[1]


def _validate(text: Optional[str], min_length: Any, max_length: Any) -> Tuple[str, int, int]:
    text = shell.require_text(text, "some text to summarize")
    words = len(text.split())
    if words < MIN_WORDS:
        shell.fail(f"Please enter at least {MIN_WORDS} words to summarize (this text has {words}).")
    if min_length is None or max_length is None:
        shell.fail("Please set both the minimum and the maximum summary length.")
    min_length, max_length = int(min_length), int(max_length)
    if min_length >= max_length:
        shell.fail(
            f"The minimum summary length ({min_length} tokens) must be smaller than the maximum ({max_length} tokens)."
        )
    return text, min_length, max_length


def _special_tokens(tokenizer: Any) -> int:
    try:
        return int(tokenizer.num_special_tokens_to_add())
    except Exception:
        return 0


def _run(
    choice: shell.ModelChoice,
    summarizer: Any,
    text: str,
    min_length: int,
    max_length: int,
    label: str = "Text",
) -> Tuple[str, Dict[str, Any]]:
    """
    Summarize `text` with one loaded pipeline.

    Applies the registry prefix unless the pipeline already adds it (T5
    checkpoints with ``task_specific_params`` do), clips the input to the
    model's limit and caps ``max_length`` at the input length so short inputs
    do not trigger the pipeline's "max_length is set to N" warning.
    """
    tokenizer = getattr(summarizer, "tokenizer", None)
    builtin = getattr(summarizer, "prefix", None) or ""
    prefix = choice.config.prefix or ""
    if prefix and builtin.strip() == prefix.strip():
        prefix = ""  # the pipeline prepends it itself
    reserve = shell.count_tokens(tokenizer, builtin + prefix) if tokenizer is not None and (builtin or prefix) else 0

    used = shell.clip_for_model(text, summarizer, label=label, reserve=reserve)
    model_input = prefix + used

    gen_min, gen_max = min_length, max_length
    if tokenizer is not None:
        input_tokens = shell.count_tokens(tokenizer, builtin + model_input) + _special_tokens(tokenizer)
        if gen_max > input_tokens:
            gen_max = input_tokens
            if gen_min >= gen_max:
                gen_min = max(1, gen_max // 2)
            shell.info(
                f"{choice.name}: the input is only {input_tokens} tokens, "
                f"so the summary is limited to {gen_min} to {gen_max} tokens."
            )

    # No truncation=True: the input is already clipped, and BART tokenizers without
    # model_max_length only log a "no maximum length" notice for it.
    start = time.perf_counter()
    output = summarizer(model_input, min_length=gen_min, max_length=gen_max, do_sample=False)
    elapsed = time.perf_counter() - start

    summary = output[0]["summary_text"].strip()
    input_words, summary_words = len(used.split()), len(summary.split())
    stats = {
        "Model": choice.model_id,
        "Input": f"{input_words} words" + (" (truncated to the model's limit)" if used != text else ""),
        "Summary": f"{summary_words} words",
        "Compression": f"{1 - summary_words / max(1, input_words):.0%} shorter",
        "Length limits (tokens)": f"{gen_min} to {gen_max}",
        "Generation time": f"{elapsed:.1f} s",
    }
    return summary, stats


# --------------------------------------------------------------------------- handlers


def summarize_text(text: str, model_name: str, min_length: float, max_length: float) -> Tuple[str, Dict[str, Any]]:
    """Summarize with one model; returns the summary and statistics."""
    text, min_length, max_length = _validate(text, min_length, max_length)
    choice = MODELS.get(model_name)
    summarizer = MODELS.load(model_name)
    try:
        return _run(choice, summarizer, text, min_length, max_length)
    except Exception as exc:
        shell.fail_from(exc, "Summarization")


def compare_models(
    text: str, model_names: Optional[List[str]], min_length: float, max_length: float
) -> Iterator[pd.DataFrame]:
    """
    Summarize with several models one after another, yielding the table after
    each model so results appear as they finish. A failing model is reported in
    its row instead of failing the whole comparison.
    """
    text, min_length, max_length = _validate(text, min_length, max_length)
    choices = [MODELS.get(name) for name in dict.fromkeys(model_names or [])]
    if not choices:
        shell.fail("Please select at least one model to compare.")

    cache_size = shell.get_settings().max_cached_models
    if len(choices) > cache_size:
        shell.warn(
            f"{len(choices)} models selected but the model cache holds {cache_size} "
            "(NLP_MAX_CACHED_MODELS). They run one at a time and evict each other, "
            "so each comparison reloads them."
        )

    rows: List[List[Any]] = []
    for choice in choices:
        start = time.perf_counter()
        try:
            summarizer = MODELS.load(choice.name)
            summary, _ = _run(choice, summarizer, text, min_length, max_length, label=f"Text for {choice.name}")
            rows.append([choice.name, summary, len(summary.split()), round(time.perf_counter() - start, 1)])
        except Exception as exc:
            message = exc.message if isinstance(exc, gr.Error) else f"{type(exc).__name__}: {exc}"
            logger.warning(f"Comparison: {choice.name} failed: {message}")
            shell.warn(f"{choice.name} failed; see its row for details.")
            rows.append([choice.name, f"Error: {message[:300]}", 0, round(time.perf_counter() - start, 1)])
        yield pd.DataFrame(rows, columns=COMPARE_COLUMNS)


def use_article(article: str) -> Tuple[str, gr.Tabs]:
    """Put an example article into the summarize box and switch to that tab."""
    return article, gr.Tabs(selected=TAB_SUMMARIZE)


# --------------------------------------------------------------------------- page

EXAMPLE_ARTICLES = {
    "Amazon rainforest": (
        "The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical "
        "rainforest in the Amazon biome that covers most of the Amazon basin of South "
        "America. This basin encompasses 7,000,000 km2 (2,700,000 sq mi), of which "
        "5,500,000 km2 (2,100,000 sq mi) are covered by the rainforest. The majority of "
        "the forest is contained within Brazil, with 60% of the rainforest, followed by "
        "Peru with 13%, Colombia with 10%, and with minor amounts in Bolivia, Ecuador, "
        "French Guiana, Guyana, Suriname, and Venezuela.\n\n"
        "The Amazon represents over half of the planet's remaining rainforests, and "
        "comprises the largest and most biodiverse tract of tropical rainforest in the "
        "world, with an estimated 390 billion individual trees divided into 16,000 species. "
        "The region is home to about 2.5 million insect species, tens of thousands of "
        "plants, and some 2,000 birds and mammals. To date, at least 40,000 plant species, "
        "2,200 fishes, 1,294 birds, 427 mammals, 428 amphibians, and 378 reptiles have been "
        "scientifically classified in the region.\n\n"
        "The rainforest likely formed during the Eocene era, following the extinction event "
        "which wiped out the dinosaurs. It appeared following a global reduction of tropical "
        "temperatures when the Atlantic Ocean had widened sufficiently to provide a warm, "
        "moist climate to the Amazon basin. The rainforest has been in existence for at "
        "least 55 million years, and most of the region remained free of savanna-type biomes "
        "during glacial periods, allowing for the survival and evolution of a broad diversity "
        "of species."
    ),
    "Artificial intelligence": (
        "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed "
        "to natural intelligence displayed by animals including humans. AI research has been "
        "defined as the field of study of intelligent agents, which refers to any system that "
        "perceives its environment and takes actions that maximize its chance of achieving "
        'its goals. The term "artificial intelligence" had previously been used to describe '
        'machines that mimic and display "human" cognitive skills that are associated with '
        'the human mind, such as "learning" and "problem-solving". This definition has '
        "since been rejected by major AI researchers who now describe AI in terms of "
        "rationality and acting rationally, which does not limit how intelligence can be "
        "articulated.\n\n"
        "AI applications include advanced web search engines, recommendation systems, "
        "understanding human speech, self-driving cars, generative or creative tools, "
        "automated decision-making and competing at the highest level in strategic game "
        "systems. As machines become increasingly capable, tasks considered to require "
        '"intelligence" are often removed from the definition of AI, a phenomenon known '
        "as the AI effect. For instance, optical character recognition is frequently excluded "
        "from things considered to be AI, having become a routine technology.\n\n"
        "Artificial intelligence was founded as an academic discipline in 1956, and in the "
        "years since it has experienced several waves of optimism, followed by disappointment "
        "and the loss of funding, followed by new approaches, success and renewed funding. "
        "AI research has tried and discarded many different approaches, including simulating "
        "the brain, modeling human problem solving, formal logic, large databases of knowledge "
        "and imitating animal behavior. In the first decades of the 21st century, highly "
        "mathematical and statistical machine learning has dominated the field, and this "
        "technique has proved highly successful, helping to solve many challenging problems "
        "throughout industry and academia."
    ),
    "Climate change": (
        "Climate change refers to long-term shifts in temperatures and weather patterns. "
        "Such shifts can be natural, due to changes in the sun's activity or large volcanic "
        "eruptions. But since the 1800s, human activities have been the main driver of "
        "climate change, primarily due to the burning of fossil fuels like coal, oil, and "
        "gas. Burning fossil fuels generates greenhouse gas emissions that act like a blanket "
        "wrapped around the Earth, trapping the sun's heat and raising temperatures.\n\n"
        "The main greenhouse gases that are causing climate change include carbon dioxide and "
        "methane. These come from using gasoline for driving a car or coal for heating a "
        "building, for example. Clearing land and cutting down forests can also release "
        "carbon dioxide. Agriculture, oil and gas operations are major sources of methane "
        "emissions. Energy, industry, transport, buildings, agriculture and land use are "
        "among the main sectors causing greenhouse gases.\n\n"
        "The consequences of climate change now include intense droughts, water scarcity, "
        "severe fires, rising sea levels, flooding, melting polar ice, catastrophic storms "
        "and declining biodiversity. People are experiencing climate change in diverse ways. "
        "It affects our health, ability to grow food, housing, safety and work. Some of us "
        "are already more vulnerable to climate impacts, such as people living in small "
        "island nations and other developing countries. Conditions like sea-level rise and "
        "saltwater intrusion have advanced to the point where whole communities have had to "
        "relocate, and protracted droughts are putting people at risk of famine. In the "
        "future, the number of climate refugees is expected to rise."
    ),
}

ABOUT = f"""
Lengths are measured in model **tokens** (roughly three quarters of a word each). Choosing a
model resets the sliders to that model's defaults. Input needs at least {MIN_WORDS} words; text
longer than the model's limit (about 1,000 tokens for BART, 512 for T5 and Pegasus) is cut and
you get a warning.

Tips: BART and DistilBART (trained on CNN/DailyMail news) write extractive-style, multi-sentence
summaries; Pegasus XSum writes one very short sentence; T5 is a general text-to-text model.
The comparison tab runs models one after another, so selecting more models than the cache
holds makes them reload on every run.
"""


def _length_sliders(defaults: Tuple[int, int]) -> Tuple[gr.Slider, gr.Slider]:
    with gr.Row():
        min_slider = gr.Slider(5, 200, value=defaults[0], step=5, label="Minimum summary length (tokens)")
        max_slider = gr.Slider(20, 512, value=defaults[1], step=10, label="Maximum summary length (tokens)")
    return min_slider, max_slider


def create_ui() -> gr.Blocks:
    """Build the summarization page."""
    first_article = next(iter(EXAMPLE_ARTICLES.values()))
    with shell.page(UI_KEY) as demo:
        with gr.Tabs(selected=TAB_SUMMARIZE) as tabs:
            with gr.Tab("Summarize", id=TAB_SUMMARIZE):
                with gr.Row():
                    with gr.Column(scale=3):
                        single_text = gr.Textbox(label="Text to summarize", value=first_article, lines=14)
                        single_model = MODELS.dropdown()
                        single_btn = gr.Button("Summarize", variant="primary")
                        single_min, single_max = _length_sliders(length_defaults(MODELS.default))
                    with gr.Column(scale=2):
                        summary_output = gr.Textbox(label="Summary", lines=10, interactive=False)
                        stats_output = gr.JSON(label="Statistics")

            with gr.Tab("Compare models", id=TAB_COMPARE):
                with gr.Row():
                    with gr.Column(scale=2):
                        compare_text = gr.Textbox(
                            label="Text to summarize",
                            value=EXAMPLE_ARTICLES["Artificial intelligence"],
                            lines=14,
                        )
                        compare_select = gr.CheckboxGroup(
                            choices=MODELS.names,
                            value=COMPARE_DEFAULT,
                            label="Models to compare",
                            info=f"Run one after another; the cache holds "
                            f"{shell.get_settings().max_cached_models} models at a time.",
                        )
                        compare_btn = gr.Button("Compare", variant="primary")
                        compare_min, compare_max = _length_sliders(FALLBACK_LENGTHS)
                    with gr.Column(scale=3):
                        comparison_output = gr.Dataframe(
                            headers=COMPARE_COLUMNS, label="Summaries", wrap=True, interactive=False
                        )

            with gr.Tab("Example articles", id=TAB_EXAMPLES):
                article_buttons = []
                for title, article in EXAMPLE_ARTICLES.items():
                    with gr.Group():
                        gr.Textbox(value=article, label=title, lines=6, interactive=False)
                        article_buttons.append((gr.Button(f"Use this article: {title}", size="sm"), article))

        shell.about(ABOUT, MODELS)

        single_btn.click(
            summarize_text,
            inputs=[single_text, single_model, single_min, single_max],
            outputs=[summary_output, stats_output],
        )
        single_model.change(
            length_defaults, inputs=single_model, outputs=[single_min, single_max], api_visibility="undocumented"
        )
        compare_btn.click(
            compare_models,
            inputs=[compare_text, compare_select, compare_min, compare_max],
            outputs=comparison_output,
        )
        for button, article in article_buttons:
            button.click(lambda a=article: use_article(a), outputs=[single_text, tabs], api_visibility="undocumented")
    return demo


if __name__ == "__main__":
    shell.launch(create_ui(), UI_KEY, argv=sys.argv[1:])
