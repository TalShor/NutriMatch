from __future__ import annotations

"""GPT-assisted comparison of food items.

This module provides a *single public helper* – ``compare_dataframe`` – that takes a
``pd.DataFrame`` whose **rows are food-item pairs** (the first and second
columns hold arbitrary JSON dictionaries describing the two items).
It returns the *same* DataFrame with two extra columns:

* ``nutritionally_equivalent`` – ``bool`` – GPT judgement whether the two
  records can be considered nutritionally exchangeable *given their main
  ingredients *and* cooking / preparation method*.

Usage example
-------------
```python
>>> import pandas as pd, json
>>> from nutrimatch_v2.src.gpt_tools.comparison import compare_dataframe
>>> df = pd.DataFrame({
...     "item_a": [json.dumps({"name": "Boiled potato", "weight_g": 150})],
...     "item_b": [json.dumps({"name": "Baked potato", "weight_g": 150})],
... })
>>> out = compare_dataframe(df)
>>> out[["nutritionally_equivalent"]]
    nutritionally_equivalent
0                     True
```

The heavy lifting is done by OpenAI GPT-4 (or a compatible model) via the
*function-calling* interface, ensuring structured machine-readable output and
robust execution (automatic retries with exponential back-off).
"""

import concurrent.futures
import json
import os
from typing import List

import openai
import pandas as pd
from pydantic import BaseModel, Field, conlist, create_model
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ---------------------------------------------------------------------------
# NEW concise system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT: dict = {
    "role": "system",
    "content": (
        "You are a dietician. For each food-item *pair*, decide whether they are **nutritionally similar** – i.e. a typical serving of one can replace the other without meaningful change (≈ ≤ 10 % difference) in calories, macronutrients, or key micronutrients.\n\n"
        "Processing state matters: raw, cooked, frozen, canned, dried, fermented, etc.  Different states are generally NOT nutritionally similar unless nutrient values still fall within the 10 % tolerance.\n\n"
        "Regard the items as the *same* when differences are purely cosmetic (macros still within the 10 % band), for example: \n"
        "• word order, punctuation, capitalisation, singular/plural, hyphenation\n"
        "• synonyms, generic category words, or small morphological variants (e.g. ‘malt’ vs ‘malted’, ‘uncooked’ vs ‘unprepared’)\n"
        "• explanatory text in parentheses or variety/cultivar notes (e.g. ‘includes crisphead types’)\n"
        "• negligible ingredients such as skin, glaze, seasoning, or ‘commercially prepared’ qualifiers\n\n"
        "DO NOT treat items as similar when the preparation/processing state changes nutrients beyond tolerance – e.g. raw vs cooked, raw vs frozen, sweetened vs unsweetened, or cocoa-solid ranges that imply >10 % change.\n\n"
        "Reply **exclusively** with the provided function call – no free text."
    ),
}


class PairJudgement(BaseModel):
    """Structured GPT output for a single food-item pair."""

    nutritionally_equivalent: bool = Field(
        ...,
        description="True when the two items are nutritionally similar as defined in the system prompt (≈ ≤ 5 % difference in energy/macros/micros).",
    )


# ---------------------------------------------------------------------------
# GPT helpers (per-batch)
# ---------------------------------------------------------------------------


def _build_prompt(batch_df: pd.DataFrame, tool_name: str) -> List[dict]:
    """Compose the chat prompt for the current batch."""

    prompt: List[dict] = [_SYSTEM_PROMPT]

    # -------------------------------------------------------------------
    # Few-shot examples to prime the model
    # -------------------------------------------------------------------
    few_shots = [
        (
            {
                "description": "granola bar, chocolate coated, with coconut",
                "food_category": "Snacks",
            },
            {
                "description": "snacks, granola bar, with coconut, chocolate coated",
                "food_category": "Snacks",
            },
            True,
        ),
        (
            {
                "description": "barley flour, malted",
                "food_category": "Cereal Grains and Pasta",
            },
            {
                "description": "barley malt flour",
                "food_category": "Cereal Grains and Pasta",
            },
            True,
        ),
        (
            {
                "description": "peas, green, raw",
                "food_category": "Vegetables and Vegetable Products",
            },
            {
                "description": "green peas, raw, frozen",
                "food_category": "Vegetables and Vegetable Products",
            },
            False,
        ),
        (
            {
                "description": "nuts, almonds, oil roasted, with salt added",
                "food_category": "Nut and Seed Products",
            },
            {
                "description": "almonds, roasted, salted",
                "food_category": "Nut and Seed Products",
            },
            True,
        ),
        (
            {
                "description": "cucumber, with peel, raw",
                "food_category": "Vegetables and Vegetable Products",
            },
            {
                "description": "cucumber, raw, without peel",
                "food_category": "Vegetables and Vegetable Products",
            },
            True,
        ),
        (
            {
                "description": "alcoholic beverage, wine, table, red",
                "food_category": "Beverages",
            },
            {"description": "wine, table, red", "food_category": "Alcoholic Beverages"},
            True,
        ),
        (
            {
                "description": "goose, domesticated, meat and skin, cooked, roasted",
                "food_category": "Poultry Products",
            },
            {
                "description": "goose, domesticated, meat and skin, raw",
                "food_category": "Poultry Products",
            },
            False,
        ),
        (
            {
                "description": "chocolate, dark, 70 85% cacao solids",
                "food_category": "Sweets",
            },
            {
                "description": "chocolate, dark, 60 69% cacao solids",
                "food_category": "Sweets",
            },
            False,
        ),
        (
            {
                "description": "cauliflower, frozen, unprepared",
                "food_category": "Vegetables and Vegetable Products",
            },
            {
                "description": "cauliflower, frozen, uncooked",
                "food_category": "Vegetables and Vegetable Products",
            },
            True,
        ),
        (
            {
                "description": "raspberries, frozen, red, sweetened",
                "food_category": "Fruits and Fruit Juices",
            },
            {
                "description": "raspberries, frozen, sweetened",
                "food_category": "Fruits and Fruit Juices",
            },
            True,
        ),
        (
            {
                "description": "bread, whole wheat, commercially prepared, toasted",
                "food_category": "Baked Products",
            },
            {
                "description": "bread, whole wheat, toasted",
                "food_category": "Baked Products",
            },
            True,
        ),
    ]

    for idx, (item_a, item_b, is_equiv) in enumerate(few_shots, start=1):
        example_payload = [{"item_a": item_a, "item_b": item_b}]
        tool_id = f"example_{idx}"

        # User message with the example pair
        prompt.append(
            {
                "role": "user",
                "content": (
                    "For each element decide nutritional similarity. JSON list:\n"
                    f"{json.dumps(example_payload, ensure_ascii=False)}"
                ),
            }
        )

        # Assistant message calling the function tool
        prompt.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(
                                {"judgements": [{"nutritionally_equivalent": is_equiv}]}
                            ),
                        },
                    }
                ],
            }
        )

        # Matching tool response message (required by OpenAI)
        prompt.append(
            {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": json.dumps(
                    {"judgements": [{"nutritionally_equivalent": is_equiv}]}
                ),
            }
        )

    # -------------------------------------------------------------------
    # Actual user batch
    # -------------------------------------------------------------------
    payload = batch_df.to_dict(orient="records")
    user_msg = {
        "role": "user",
        "content": (
            "For every element in the JSON list decide if *item_a* and *item_b*\n"
            "are nutritionally similar (see guidelines above).  Reply ONLY via the tool call.\n\n"
            f"JSON list:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        ),
    }

    prompt.append(user_msg)
    return prompt


@retry(
    retry=retry_if_exception_type(Exception),  # on *any* error
    stop=stop_after_attempt(3),  # up to 3 tries
    wait=wait_exponential(multiplier=2, min=1, max=30),  # 1 s → 2 s → 4 s (capped)
    reraise=True,
)
def _query_gpt(batch_df: pd.DataFrame) -> pd.DataFrame:
    """Send one batch to GPT and parse the structured response."""

    batch_size = len(batch_df)

    # Build a dynamic model enforcing the *exact* list length for added safety
    BatchModel = create_model(
        "BatchPairJudgements",
        judgements=(
            conlist(PairJudgement, min_length=batch_size, max_length=batch_size),
            ...,  # mandatory field
        ),
    )

    tool_spec = openai.pydantic_function_tool(BatchModel)
    tool_name = tool_spec["function"]["name"]

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",  # small, cost-effective model – adjust as needed
        messages=_build_prompt(batch_df, tool_name),
        tools=[tool_spec],
        tool_choice="auto",
    )

    # Extract the *raw* JSON arguments string
    try:
        tool_call = response.choices[0].message.tool_calls[0]
        raw_args: str = tool_call.function.arguments  # type: ignore[attr-defined]
    except (AttributeError, IndexError):
        raise RuntimeError("GPT response missing tool-call arguments")

    parsed = BatchModel(**json.loads(raw_args))
    judging_df = pd.DataFrame([j.model_dump() for j in parsed.judgements])

    # Preserve original index for alignment
    judging_df.index = batch_df.index
    return judging_df


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def compare_dataframe(
    df: pd.DataFrame,
    col_item_a: str | None = None,
    col_item_b: str | None = None,
    batch_size: int = 30,
    num_threads: int = 8,
) -> pd.DataFrame:
    """Add GPT nutritional-equivalence judgement to *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe – **must contain two columns** with JSON-serialisable
        objects representing the two food items to compare.  If *col_item_a*
        and *col_item_b* are omitted, the first two columns are assumed.
    col_item_a / col_item_b : str, optional
        Names of the columns holding the items.  Useful when *df* has more than
        two columns.
    batch_size : int
        Max number of rows sent to GPT in a single request.
    num_threads : int
        Parallelism level for batching.

    Returns
    -------
    pd.DataFrame
        The input dataframe *augmented* with ``nutritionally_equivalent``
        (bool) column.
    """

    if col_item_a is None or col_item_b is None:
        col_item_a, col_item_b = df.columns[:2]

    work_df = df[[col_item_a, col_item_b]].copy()

    # Ensure each cell is a *native* Python object (not string) to avoid double-encoding
    def _ensure_obj(val):
        return val if not isinstance(val, str) else json.loads(val)

    work_df[col_item_a] = work_df[col_item_a].apply(_ensure_obj)
    work_df[col_item_b] = work_df[col_item_b].apply(_ensure_obj)

    # Split into batches
    batches = [
        work_df.iloc[i : i + batch_size] for i in range(0, len(work_df), batch_size)
    ]
    if not batches:
        return df  # nothing to do

    # Parallel execution – thread-pool is OK because the heavy lifting is I/O bound
    results: List[pd.DataFrame] = []

    def _run(batch: pd.DataFrame):
        print(f"[GPT-compare] rows {batch.index.min()}–{batch.index.max()}")
        return _query_gpt(batch)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_map = {
            executor.submit(_run, batch): idx for idx, batch in enumerate(batches)
        }
        for fut in concurrent.futures.as_completed(future_map):
            try:
                results.append(fut.result())
            except Exception as exc:
                idx = future_map[fut]
                raise RuntimeError(f"Failed GPT comparison for batch {idx}") from exc

    comparison_df = pd.concat(results).sort_index()

    # Merge-back preserving original row order
    out = df.copy()
    out["nutritionally_equivalent"] = comparison_df["nutritionally_equivalent"]
    return out
