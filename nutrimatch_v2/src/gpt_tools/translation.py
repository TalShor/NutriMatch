import concurrent.futures
import json
import os
import pathlib
from itertools import count

import openai
import pandas as pd
import yaml
from pydantic import conlist, create_model
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..FCDBs.SR_legacy.dataset_structure import SR_LegacyFoodItem

system_prompt = {
    "role": "system",
    "content": (
        "You are a food-taxonomy expert specialising in cross-database mappings of food items.\n"
        "For each origin item, translate it to the single most likely counterpart in the target "
        "food-composition database based on descriptive features, aiming for nutritional values "
        "that are as closely aligned as possible while representing the same real-world food.\n"
        "Return the mapping using the tool below."
    ),
}


def _parse_few_shot_yaml(yaml_file_path: str, tool_name: str) -> list[dict]:
    """Convert a few-shot YAML file into an OpenAI-style message list.

    Each YAML *example* is expected in the following shape::

        example_n:
          <dataset_name>:
            origin:
              english_name: "…"
              hebrew_name: "…"
            target:
              description: "…"
              food_category: "…"
              common_name: "…"

    For every example two messages are produced:

    1. User message – the *origin* block, serialised to JSON.
    2. Assistant message – a "tool-call" message with the *target* mapping.

    The function is deliberately generic – it doesn’t hard-code the dataset
    name and will use whatever key appears (``sr_legacy`` in our current
    data).  The assistant messages use incremental ``tool_call_id`` values
    starting from ``t1`` and the *tool* name placeholder ``tool.name`` so the
    caller can patch the actual tool name later on.
    """

    yaml_path = pathlib.Path(yaml_file_path)
    if not yaml_path.is_file():
        raise FileNotFoundError(yaml_file_path)

    examples = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    messages: list[dict] = []
    id_counter = count(1)

    for key in sorted(examples.keys()):  # example_1, example_2 …
        example_block: dict = examples[key]

        # Pull whatever dataset name appears inside the example (e.g. sr_legacy)
        dataset_name = next(iter(example_block))
        data = example_block[dataset_name]

        origin = data.get("origin", {})
        target = data.get("target", {})

        # USER message – raw origin information
        messages.append(
            {
                "role": "user",
                "content": json.dumps(origin, ensure_ascii=False),
            }
        )

        # ASSISTANT tool-call message – expected mapping
        tool_call_id = f"t{next(id_counter)}"
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_call_id": tool_call_id,
                "name": tool_name,  # placeholder; replace as needed
                "arguments": json.dumps(target, ensure_ascii=False),
            }
        )

    return messages


def _get_batch_translation_prompt(
    batch_food_items: pd.DataFrame, yaml_file_path: str, tool_name: str
) -> list[dict]:
    """Get the full prompt for the translation tool."""
    batch_food_items_json = json.dumps(
        batch_food_items.to_dict(orient="records"), ensure_ascii=False, indent=2
    )
    return [
        system_prompt,
        *_parse_few_shot_yaml(yaml_file_path, tool_name),
        {
            "role": "user",
            "content": f"Convert these food item to the given format:\n{batch_food_items_json}\n. Reply ONLY via the tool.",
        },
    ]


@retry(  # <-- add resiliency / automatic retries
    retry=retry_if_exception_type(
        Exception
    ),  # retry on *any* exception (adjust if needed)
    reraise=True,  # propagate the last exception after giving up
    stop=stop_after_attempt(3),  # max 3 attempts
    wait=wait_exponential(
        multiplier=2, min=1, max=30
    ),  # 1s, 2s, 4s back-off (capped at 30s)
)
def get_batch_translation(
    batch_food_items: pd.DataFrame, yaml_file_path: str
) -> pd.DataFrame:
    # Get the food item type
    batch_size = len(batch_food_items)
    MapManyFoods = create_model(
        "MapManyFoods",
        mappings=(
            conlist(SR_LegacyFoodItem, min_length=batch_size, max_length=batch_size),
            ...,
        ),
    )

    tool_spec = openai.pydantic_function_tool(MapManyFoods)
    tool_name = tool_spec["function"]["name"]

    # Use the OpenAI v1 client interface
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    gpt_response = client.chat.completions.create(
        model="o4-mini-2025-04-16",
        messages=_get_batch_translation_prompt(
            batch_food_items, yaml_file_path, tool_name
        ),
        tools=[tool_spec],
        tool_choice="auto",
    )

    # --- Parse the tool call output into SR_LegacyFoodItem objects ---
    # The assistant should reply with a single tool call whose `arguments` JSON
    # matches the `MapManyFoods` model we defined above.  The OpenAI Python
    # client keeps the *raw* arguments payload as a JSON-formatted string.

    try:
        tool_call = gpt_response.choices[0].message.tool_calls[0]
        # In the v1 OpenAI SDK, the function arguments live under
        # `tool_call.function.arguments`.
        arguments_json: str = tool_call.function.arguments  # type: ignore[attr-defined]
    except (AttributeError, IndexError):
        raise RuntimeError(
            "The GPT response did not include the expected tool call with arguments."
        )

    # Convert the JSON string to Python dict and validate against our model
    arguments_dict = json.loads(arguments_json)

    # Parse and validate using the same Pydantic model we registered with the tool
    parsed_result = MapManyFoods(**arguments_dict)
    parsed_result_df = SR_LegacyFoodItem.fooditems2df(parsed_result.mappings)

    return parsed_result_df


def get_translation(
    few_shots_path: str,
    data: pd.DataFrame,
    batch_size: int,
    num_threads: int = 16,
) -> pd.DataFrame:

    data = data.head(30)

    # Prepare the batches
    batches = [data.iloc[i : i + batch_size] for i in range(0, len(data), batch_size)]
    if not batches:
        return data  # Nothing to translate

    # Helper for the executor
    def _translate(batch_df: pd.DataFrame) -> pd.DataFrame:
        return get_batch_translation(batch_df, few_shots_path)

    translations: list[pd.DataFrame] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_index = {
            executor.submit(_translate, batch): idx for idx, batch in enumerate(batches)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                translations.append(future.result())
            except Exception as exc:
                raise RuntimeError(f"Translation failed for batch {idx}") from exc

    translations = pd.concat(translations, ignore_index=True)
    translations["food_category"] = translations["food_category"].apply(
        lambda e: e.value
    )
    translations = SR_LegacyFoodItem.add_fcdb_to_columns(translations)

    translations.index = data.index
    # Concat the translations with the original data in the same rows - leave the data index
    data_with_translation = pd.concat([data, translations], axis=1)
    return data_with_translation
