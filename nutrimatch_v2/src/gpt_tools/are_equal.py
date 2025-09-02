"""GPT-assisted equivalence checking of food items with exact matching pre-filter.

This module provides a public helper – ``are_equal_dataframe`` – that takes a
``pd.DataFrame`` whose **rows contain a reference food item and candidates** to compare.
It returns the *same* DataFrame with an additional column:

* ``is_equivalent`` – ``List[bool]`` – GPT judgement whether each candidate
  is nutritionally equivalent to the reference item *given their main
  ingredients *and* cooking / preparation method*.

Usage example
-------------
```python
>>> import pandas as pd, json
>>> from nutrimatch_v2.src.gpt_tools.are_equal import are_equal_dataframe
>>> df = pd.DataFrame({
...     "reference": [json.dumps({"name": "Boiled potato", "weight_g": 150})],
...     "candidates": [[
...         json.dumps({"name": "Baked potato", "weight_g": 150}),
...         json.dumps({"name": "French fries", "weight_g": 150}),
...         json.dumps({"name": "Mashed potato", "weight_g": 150})
...     ]],
... })
>>> out = are_equal_dataframe(df)
>>> out[["is_equivalent"]]
    is_equivalent
0  [True, False, True]
```

The heavy lifting is done by OpenAI GPT-4 (or a compatible model) via the
*function-calling* interface, ensuring structured machine-readable output and
robust execution (automatic retries with exponential back-off).
"""

from __future__ import annotations

import concurrent.futures
import json
import os
from ast import literal_eval
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

from ..syntax_matching.exact_matcher import FoodDescriptionNormalizer

# ---------------------------------------------------------------------------
# helper – check equivalence of candidates vs. reference
# ---------------------------------------------------------------------------

# System prompt for *equivalence* task (reference vs. candidates)
_SYSTEM_PROMPT_EQUIVALENCE = {
    "role": "system",
    "content": (
        "For each reference food item, determine which candidates have the same ingredients and same cooking method.\n\n"
        "Return True only if the candidate has identical main ingredients and identical cooking method.\n"
        "Return False if ingredients differ or cooking method differs.\n"
        "Use common sense for unspecified cooking methods (e.g., 'apple' = raw apple).\n\n"
        "Use the function call format to return your answers."
    ),
}


class EquivalenceResult(BaseModel):
    """Structured GPT output: list of boolean equivalence judgments."""

    is_equivalent: List[bool] = Field(
        ...,
        description=(
            "List of boolean values indicating whether each candidate is "
            "nutritionally equivalent to the reference item."
        ),
    )


# ---------------------------------------------------------------------------
# GPT helpers (per-batch) – equivalence task
# ---------------------------------------------------------------------------


def _build_equivalence_prompt(batch_df: pd.DataFrame, tool_name: str) -> List[dict]:
    """Compose chat prompt for *equivalence* batches."""

    prompt: List[dict] = [_SYSTEM_PROMPT_EQUIVALENCE]

    # Apply text normalization to the input data
    normalizer = FoodDescriptionNormalizer()

    def normalize_food_item(item):
        """Normalize a food item description."""
        if isinstance(item, dict):
            if "description" in item:
                item = item.copy()
                item["description"] = normalizer.normalize(item["description"])
            return item
        elif isinstance(item, str):
            try:
                # Try to parse as dict if it's a string representation
                parsed = eval(item)
                if isinstance(parsed, dict) and "description" in parsed:
                    parsed["description"] = normalizer.normalize(parsed["description"])
                    return parsed
            except (ValueError, SyntaxError, TypeError):
                pass
            return normalizer.normalize(item)
        return item

    # Normalize the batch data
    normalized_data = []
    for _, row in batch_df.iterrows():
        normalized_row = {}
        for col, value in row.items():
            if isinstance(value, list):
                # Normalize each item in the list (candidates)
                normalized_row[col] = [normalize_food_item(item) for item in value]
            else:
                # Normalize single item (reference)
                normalized_row[col] = normalize_food_item(value)
        normalized_data.append(normalized_row)

    # Create the user prompt
    prompt.append(
        {
            "role": "user",
            "content": (
                f"JSON list:\n{json.dumps(normalized_data, ensure_ascii=False, indent=2)}"
            ),
        }
    )

    return prompt


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=1, max=30),
    reraise=True,
)
def _query_equivalence_gpt(batch_df: pd.DataFrame) -> pd.DataFrame:
    """Send one *equivalence* batch to GPT and parse the response."""

    batch_size = len(batch_df)

    # Dynamic model enforcing exact list length
    BatchModel = create_model(
        "BatchEquivalenceResults",
        equivalences=(
            conlist(EquivalenceResult, min_length=batch_size, max_length=batch_size),
            ...,
        ),
    )

    tool_spec = openai.pydantic_function_tool(BatchModel)
    tool_name = tool_spec["function"]["name"]

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=_build_equivalence_prompt(batch_df, tool_name),
        tools=[tool_spec],
        tool_choice="auto",
    )

    # Extract raw JSON arguments
    try:
        # save the response to a file
        tool_call = response.choices[0].message.tool_calls[0]

        # Convert batch_df to records for including input data
        input_data = batch_df.to_dict(orient="records")

        # Convert tool_call to a serializable format
        tool_call_dict = {
            "id": tool_call.id,
            "type": tool_call.type,
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            },
        }

        # Parse the GPT response to get equivalence results
        parsed_response = json.loads(tool_call.function.arguments)
        equivalences = parsed_response["equivalences"]

        # Create simple response with just the boolean arrays for debugging
        boolean_results = []
        for i, row in enumerate(input_data):
            is_equivalent_list = equivalences[i]["is_equivalent"]
            boolean_results.append(is_equivalent_list)

        # Create simple response with just the boolean arrays
        comprehensive_response = {
            "equivalence_results": boolean_results,
            "batch_size": len(batch_df),
            "full_gpt_response": tool_call_dict,
            "verification_info": "equivalence_results[i] contains the boolean array for input row i",
        }

        with open("gpt_response.json", "w") as f:
            json.dump(comprehensive_response, f, indent=2)

        raw_args: str = tool_call.function.arguments  # type: ignore[attr-defined]
    except (AttributeError, IndexError):
        raise RuntimeError("GPT response missing tool-call arguments")

    parsed = BatchModel(**json.loads(raw_args))
    equivalences_df = pd.DataFrame([eq.model_dump() for eq in parsed.equivalences])

    # Create a copy of the original batch_df and add the equivalence results
    result_df = batch_df.copy()
    result_df["is_equivalent"] = equivalences_df["is_equivalent"].tolist()

    # Preserve index alignment
    result_df.index = batch_df.index
    return result_df


# ---------------------------------------------------------------------------
# Public helper – equivalence checking
# ---------------------------------------------------------------------------


def are_equal_dataframe(
    df: pd.DataFrame,
    col_reference: str | None = None,
    col_candidates: str | None = None,
    batch_size: int = 30,
    num_threads: int = 8,
) -> pd.DataFrame:
    """Check nutritional equivalence between reference items and their candidates.

    The dataframe is expected to contain one row per reference item, with
    the reference in *col_reference* and a list of candidates in *col_candidates*.
    The function asks GPT to determine which candidates are nutritionally
    equivalent to the reference and returns the dataframe with an additional
    ``is_equivalent`` column containing a list of boolean values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with reference items and candidate lists
    col_reference : str, optional
        Column name containing reference items (defaults to first column)
    col_candidates : str, optional
        Column name containing candidate lists (defaults to second column)
    batch_size : int, default 30
        Number of rows to process in each GPT batch
    num_threads : int, default 8
        Number of parallel threads for GPT processing

    Returns
    -------
    pd.DataFrame
        Input dataframe with additional ``is_equivalent`` column
    """

    if col_reference is None or col_candidates is None:
        col_reference, col_candidates = df.columns[:2]

    # Helper to ensure dictionary objects (not JSON strings)
    def _ensure_obj(val):
        return val if not isinstance(val, str) else json.loads(val)

    # Prepare the data for GPT processing
    # TODO: make this a group by and then do the exact matching
    work_df = (
        df.copy()
        .astype({col_reference: str, col_candidates: str})
        .groupby(col_reference)[col_candidates]
        .apply(list)
        .to_frame()
        .reset_index()
    )

    if len(work_df) > 0:
        # Split GPT rows into batches
        batches = [
            work_df.iloc[i : i + batch_size] for i in range(0, len(work_df), batch_size)
        ]

        # Parallel GPT execution
        results: List[pd.DataFrame] = []

        def _run(batch: pd.DataFrame):
            print(f"[GPT-equivalence] rows {batch.index.min()}–{batch.index.max()}")
            return _query_equivalence_gpt(batch)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_map = {
                executor.submit(_run, batch): idx for idx, batch in enumerate(batches)
            }
            for fut in concurrent.futures.as_completed(future_map):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    idx = future_map[fut]
                    raise RuntimeError(
                        f"Failed GPT equivalence check for batch {idx}"
                    ) from exc

        # Concatenate all batch results
        combined_results = pd.concat(results, ignore_index=False)

        # Validate that we haven't lost any data
        original_count = len(work_df)
        result_count = len(combined_results)
        if original_count != result_count:
            raise RuntimeError(
                f"Data loss detected: original {original_count} rows, result {result_count} rows"
            )

        # Check that all original indices are preserved
        original_indices = set(work_df.index)
        result_indices = set(combined_results.index)
        if original_indices != result_indices:
            missing_indices = original_indices - result_indices
            extra_indices = result_indices - original_indices
            raise RuntimeError(
                f"Index mismatch: missing {missing_indices}, extra {extra_indices}"
            )

        # Process the combined results to expand candidates
        work_df = (
            combined_results.set_index(col_reference)
            .apply(
                lambda row: list(zip(row[col_candidates], row["is_equivalent"])),
                axis=1,
            )
            .explode()
            .apply(pd.Series)
            .rename(columns={0: col_candidates, 1: "is_equivalent"})
            .reset_index()
            .assign(**{col_reference: lambda df: df[col_reference].apply(literal_eval)})
            .assign(
                **{col_candidates: lambda df: df[col_candidates].apply(literal_eval)}
            )
        )

        # Add embedding similarity if it exists in original df
        if "embedding_similarity" in df.columns:
            work_df["embedding_similarity"] = df["embedding_similarity"].values

        return work_df

    return df
