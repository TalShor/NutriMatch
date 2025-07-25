from __future__ import annotations

"""GPT-assisted comparison of food items.

This module provides a *single public helper* – ``compare_dataframe`` – that takes a
``pd.DataFrame`` whose **rows are food-item pairs** (the first and second
columns hold arbitrary JSON dictionaries describing the two items).
It returns the *same* DataFrame with two extra columns:

* ``nutritionally_equivalent`` – ``bool`` – GPT judgement whether the two
  records can be considered nutritionally exchangeable *given their main
  ingredients *and* cooking / preparation method*.
* ``equivalence_reason`` – ``str`` – a short human-readable justification.

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
>>> out[["nutritionally_equivalent", "equivalence_reason"]]
   nutritionally_equivalent                equivalence_reason
0                     True  Very similar macronutrient profile
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
# Prompt engineering helpers
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT: dict = {
    "role": "system",
    "content": (
        "You are a registered dietician specialising in food-composition tables\n"
        "and culinary techniques.  When given *pairs* of food-item records you\n"
        "must decide whether the two foods are **nutritionally equivalent**,\n"
        "meaning a typical serving of one can replace the other in a diet plan\n"
        "without materially changing energy, macronutrient balance, or major\n"
        "micronutrients.  Consider not only ingredients but also cooking or\n"
        "processing method (fried vs. boiled, raw vs. roasted, etc.).\n\n"
        "Return your judgement *exclusively* via the provided tool call; do NOT\n"
        "output any free-text.  Base your decision on common dietetic\n"
        "knowledge – no sourcing is required in the answer."
    ),
}


class PairJudgement(BaseModel):
    """Structured GPT output for a single food-item pair."""

    nutritionally_equivalent: bool = Field(
        ...,
        description="True if the two items can be swapped without changing the overall nutritional profile in most dietetic contexts.",
    )


# ---------------------------------------------------------------------------
# GPT helpers (per-batch)
# ---------------------------------------------------------------------------


def _build_prompt(batch_df: pd.DataFrame, tool_name: str) -> List[dict]:
    """Compose the chat prompt for the current batch."""

    # Serialise the records as a list of dicts -> [{"item_a": {...}, "item_b": {...}}, …]
    payload = batch_df.to_dict(orient="records")
    user_msg = {
        "role": "user",
        "content": (
            "For every element in the JSON list decide if *item_a* and *item_b*\n"
            "are nutritionally equivalent.  Reply ONLY via the tool call.\n\n"
            f"JSON list:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        ),
    }
    return [_SYSTEM_PROMPT, user_msg]


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
        model="gpt-4o-mini",  # small, cost-effective model – adjust as needed
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
        (bool) and ``equivalence_reason`` (str) columns.
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
    out["equivalence_reason"] = comparison_df["reason"]
    return out
