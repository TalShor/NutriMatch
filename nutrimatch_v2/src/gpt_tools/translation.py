import concurrent.futures
import json
import os
import pathlib
from itertools import count
from pathlib import Path

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
        "You are an expert on the USDA SR Legacy food-composition database and international food taxonomy.\n"
        "Your task is to map every *origin* food record (what ever the language) to the format resembling the USDA SR Legacy FCDB.\n"
        "\n"
        "Guidelines:\n"
        "• Write an English description in the USDA SR Legacy *style* – i.e. include cut, preparation method, fat level, seasoning, etc. – but it does *not* have to be an exact string from the database.\n"
        '• NEVER include brand names or marketing terms (e.g. "Elite", "Tnuva", "McDonald’s").\n'
        "• Select the entry whose nutritional profile and real-world characteristics are the closest match.\n"
        "• Populate the fields in the SR_LegacyFoodItem model.\n"
        "\n"
        "Naming rules (follow ALL):\n"
        "  1. Capitalise the first word and every proper noun; keep generic food words lowercase unless starting the name.\n"
        "  2. Order: base food → key qualifier(s) → extras. Example: 'Cheddar cheese (cow’s milk), 32 % fat, natural'.\n"
        "  3. Put clarifications in parentheses.\n"
        "  4. Separate attributes with commas; use 'and' only before the last item.\n"
        "  5. Cooking state goes last, preferably in parentheses. Use '(raw)' for uncooked.\n"
        "  6. Replace translation artefacts like 'ns as to …' with clear English ('part not specified').\n"
        "  7. Prefer standard culinary English terms (whole milk, sirloin, etc.).\n"
        "  8. Numbers: no space before %, write '3 % fat'; vitamins in capitals (B12, D, E).\n"
        "  9. Use correct singular/plural forms.\n"
        "  10. Avoid redundancy (e.g. 'boneless' already implies 'without bone').\n"
        "  11. Skip brand names; shorten long flavour lists to 'assorted fruit flavours'.\n"
        "  12. Proofread: collapse multiple spaces, trim trailing spaces, ensure readability.\n"
        "  13. Use 'homemade' (or similar) ONLY when the source explicitly indicates it. Otherwise leave it out.\n"
        "  14. Keep the original primary ingredient/species – never replace with an unrelated U.S. substitute. But you can replace the food item name if it's analagous to a US food item.\n"
        "  15. Remove vague packaging words ('pack', 'package', 'jar') unless nutritionally relevant (e.g. 'oil-packed tuna').\n"
        "  16. Assume items are commercial unless 'homemade' is explicit; use 'commercial' only when contrasting with 'homemade'.\n"
        "  17. If the food item is not a food item, set the unlikely_food_item flag to True.\n"
        "Return the result exclusively via the provided tool call; DO NOT output any free text."
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
        model="gpt-4.1-mini-2025-04-14",
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
    temp_dir: str = "temp",
) -> pd.DataFrame:

    # shouldn't have an index
    data = data.reset_index(drop=True)

    # Prepare the batches
    batches = [data.iloc[i : i + batch_size] for i in range(0, len(data), batch_size)]
    if not batches:
        return data  # Nothing to translate

    # Helper for the executor
    def _translate(batch_df: pd.DataFrame) -> pd.DataFrame:
        print(f"Translating batch {batch_df.index.min()} to {batch_df.index.max()}")
        temp_file_path = os.path.join(
            temp_dir, f"batch_{batch_df.index.min()}_{batch_df.index.max()}.parquet"
        )
        if not os.path.exists(temp_file_path):
            os.makedirs(temp_dir, exist_ok=True)
            try:
                df = get_batch_translation(batch_df, few_shots_path)
                df["food_category"] = df["food_category"].apply(lambda e: e.value)
                df.to_parquet(temp_file_path)
                return df
            except Exception as e:
                raise Exception(
                    f"Error translating batch {batch_df.index.min()} to {batch_df.index.max()} - {e}"
                )
        else:
            return pd.read_parquet(temp_file_path)

    # Run the batch translations concurrently instead of sequentially

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

    def _add_batch_index(p: Path) -> tuple[int, int]:
        try:
            df = pd.read_parquet(p)
            _, start, end = str(p.name).split(".")[0].split("_")
            df.index = range(int(start), int(end) + 1)
        except Exception as e:
            print(
                f"Error reading parquet file {p} - {e}: len(df) = {len(df)}, len(range) = {len(range(int(start), int(end) + 1))}"
            )
            raise e
        return df

    translations = [_add_batch_index(p) for p in Path(temp_dir).glob("*.parquet")]
    translations = pd.concat(translations)
    translations = SR_LegacyFoodItem.add_fcdb_to_columns(translations)

    # Concat the translations with the original data in the same rows - leave the data index
    data_with_translation = pd.concat([data, translations], axis=1)
    return data_with_translation
