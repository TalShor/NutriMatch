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

from .exact_matcher import ExactFoodMatcher

# ---------------------------------------------------------------------------
# helper – check equivalence of candidates vs. reference
# ---------------------------------------------------------------------------

# System prompt for *equivalence* task (reference vs. candidates)
_SYSTEM_PROMPT_EQUIVALENCE = {
    "role": "system",
    "content": (
        "You are a certified dietitian specialising in food substitution.\n\n"
        "INPUT (per element):\n"
        "• reference – a food description\n"
        "• candidates – 1…N alternative descriptions\n"
        "OUTPUT: Return ONLY the function-call with "
        '{ "equivalences": [ { "is_equivalent": [bool, bool, ...] } … ] }.\n'
        "Each boolean indicates whether the corresponding candidate is nutritionally equivalent to the reference.\n\n"
        "----------------------------------------------------------------------\n"
        "EQUIVALENCE CRITERIA (prioritize exact matches, then strict rules)\n"
        "0. EXACT MATCH PRIORITY – If descriptions refer to identical foods with only trivial wording differences (word order, singular/plural, synonyms), return True immediately. Examples:\n"
        "   • 'almonds, dry roasted, salted' ≈ 'nuts, almonds, dry roasted, with salt added' = TRUE\n"
        "   • 'blanched almonds' ≈ 'nuts, almonds, blanched' = TRUE\n"
        "   • 'basil, fresh' ≈ 'basil, fresh' = TRUE\n"
        "1. Core Food Identity – EXACT SAME primary food (species/plant/animal, cut). NO cross-species: beets≠radishes, blueberries≠strawberries, asparagus≠broccoli, brazil nuts≠cashews.\n"
        "2. Processing State Sensitivity:\n"
        "   • Raw ≠ cooked (significant nutritional differences)\n"
        "   • Sweetened ≠ unsweetened (significant calorie differences)\n"
        "   • Fresh ≈ frozen (if same state), but ≠ canned ≠ dried ≠ pickled\n"
        "   • Similar cooking methods: boiled≈steamed≈poached for same food\n"
        "3. Minor Additions – Salt, herbs, spices in small amounts are equivalent. 'With salt' ≈ 'without salt' for same food.\n"
        "4. Food Group – Must be same USDA Food Group.\n"
        "5. Threshold – Return True ONLY if ALL criteria satisfied. When uncertain, return False.\n\n"
        "----------------------------------------------------------------------\n"
        "SCENARIO-SPECIFIC GUIDELINES (augment rules above)\n"
        "• Plant-based milks vs. dairy milk: treat fortified soy/almond/oat milks "
        "as equivalent to low-fat or skim cow milk if macros fit.\n"
        "• Low-sugar or sugar-free desserts: sucrose↔sweetener swap is acceptable "
        "if calories stay within ±15 %.\n"
        "• Gluten-free baked goods: GF bread/crackers may be equivalent to wheat "
        "variants as long as grain base and macros align (e.g., rice bread vs. "
        "white bread).\n"
        "• Composite dishes: All MAIN ingredients + cooking method must match; "
        "sauces or herbs may differ. E.g., 'baked salmon with lemon' ≈ 'baked "
        "salmon, plain'. Treat 'ready-to-eat' vs. 'raw ingredient' as different "
        "states.\n"
        "• Fermented dairy: yogurt ↔ kefir ↔ labneh may be equivalent if fat% and "
        "added sugar comparable.\n"
        "• Cheese fat levels: 5 % vs. 8 % is acceptable; 5 % vs. 30 % is not.\n"
        "• Oils and spreads: Avocado oil ↔ olive oil but NOT ↔ butter.\n"
        "• Fortified juices: presence of added vitamins is ignored if macros match.\n"
        "• Meat form: SAME muscle-cut or slice is equivalent; ground/minced or cured are different unless specifically similar cuts.\n"
        "• Species priority: Different species are NEVER equivalent (banana≠plantain, beef≠turkey, brazil nuts≠cashews).\n"
        "• Processing sensitivity: Sweetened≠unsweetened, cooked≠raw, chips≠raw fruit.\n"
        "• Exact word matching: Focus on food substance, not packaging language.\n\n"
        "----------------------------------------------------------------------\n"
        "ALIAS / SYNONYM CHEAT-SHEET (non-exhaustive; treat as SAME food for "
        "rules 1-3)\n"
        "— Cheeses —\n"
        "  Kashkaval → Gouda/Edam/Cheddar family (semi-hard yellow cheese)\n"
        "  Pecorino, Manchego → hard/semi-hard sheep cheese (Parmesan-like)\n"
        "  Halloumi → brined grilling cheese (close to Feta in salt/fat)\n"
        "  Tzfatit, 'salty cheese', 'white cheese' → brined fresh cheese (≈ Feta)\n"
        "  Labneh → strained yogurt / cream-cheese style spread\n"
        "  Port de Salut → semi-soft cow cheese (like Muenster)\n"
        "— Milks & Yogurts —\n"
        "  Almond/oat/soy milk → plant milk category (use fat-free or low-fat milk "
        "as proxy when macros match)\n"
        "  'Producer milk' → whole cow milk\n"
        "— Breads & Baked goods —\n"
        "  'Yellow bread', 'light rye' → rye/wheat bread family\n"
        "  'Ma'amoul' → date-filled shortbread cookie\n"
        "  'Gluten-free flour mix' → baking flour substitute (rice/corn/potato).\n"
        "— Meats & Fish —\n"
        "  'Seabass', 'sea bass' same fish; raw vs cooked are DIFFERENT states.\n"
        "  'Mosht' fish → tilapia/cichlid family.\n"
        "  Beef ≠ turkey ≠ chicken ≠ pork (different animals)\n"
        "  Different fish species are NOT equivalent (salmon ≠ trout ≠ bass)\n"
        "— Condiments & Herbs —\n"
        "  Za'atar → thyme/oregano/sumac blend\n"
        "  Amba → pickled mango sauce\n"
        "  'Asian sauce (generic)' ↔ soy-based stir-fry sauce if macros similar.\n"
        "— Processed & Cured Meats —\n"
        "  Poultry pastrami → pastrami, turkey | chicken\n"
        "— Beef Cuts —\n"
        "  Entrecôte → rib-eye steak, boneless beef rib\n"
        "— Misc —\n"
        "  'Yellow cheese' (without spec) → semi-hard cow cheese\n"
        "  'Energy bar' ↔ cereal/protein bar; compare macros not brand.\n\n"
        "----------------------------------------------------------------------\n"
        "CRITICAL RULES (in order of importance):\n"
        "1. EXACT MATCHES: Identical foods with wording differences = TRUE\n"
        "2. SPECIES RULE: Different species/varieties = FALSE (beets≠radishes, blueberries≠strawberries)\n"
        "3. PROCESSING RULE: Major processing differences = FALSE (raw≠cooked, sweetened≠unsweetened)\n"
        "4. CONSERVATIVE APPROACH: When uncertain, return False\n"
        "5. ONLY return True for nutritionally similar foods with minimal differences\n\n"
        "COMMON EXACT MATCHES:\n"
        "• 'almonds, roasted, salted' = 'nuts, almonds, roasted, with salt added'\n"
        "• 'banana, raw' = 'bananas, raw'\n"
        "• 'beets, raw' = 'beets, fresh'\n"
        "• Food name with/without 'nuts,' prefix are the same food\n"
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

    # ---------------- Few-shot examples to prime the model -----------------
    few_shots = [
        (
            {
                "description": "almonds, dry roasted, salted",
                "food_category": "Nut and Seed Products",
            },
            [
                {
                    "description": "nuts, almonds, dry roasted, with salt added",
                    "food_category": "Nut and Seed Products",
                },
                {
                    "description": "almonds, raw",
                    "food_category": "Nut and Seed Products",
                },
                {
                    "description": "cashews, roasted, salted",
                    "food_category": "Nut and Seed Products",
                },
            ],
            [
                True,
                False,
                False,
            ],  # EXACT MATCH despite wording, different processing state, different nut species
        ),
        (
            {
                "description": "blueberries, frozen, unsweetened",
                "food_category": "Fruits and Fruit Juices",
            },
            [
                {
                    "description": "blueberries, frozen, sweetened",
                    "food_category": "Fruits and Fruit Juices",
                },
                {
                    "description": "blueberries, raw",
                    "food_category": "Fruits and Fruit Juices",
                },
                {
                    "description": "blackberries, frozen, unsweetened",
                    "food_category": "Fruits and Fruit Juices",
                },
            ],
            [
                False,
                False,
                False,
            ],  # sweetened≠unsweetened, raw≠frozen, different berry species
        ),
        (
            {
                "description": "beef, ground, raw",
                "food_category": "Beef Products",
            },
            [
                {
                    "description": "beef, grass fed, ground, raw",
                    "food_category": "Beef Products",
                },
                {
                    "description": "beef, ground, cooked",
                    "food_category": "Beef Products",
                },
                {
                    "description": "turkey, ground, raw",
                    "food_category": "Poultry Products",
                },
            ],
            [
                True,
                False,
                False,
            ],  # same food different source, raw≠cooked, different animal species
        ),
        (
            {
                "description": "beets, raw",
                "food_category": "Vegetables and Vegetable Products",
            },
            [
                {
                    "description": "beets, fresh",
                    "food_category": "Vegetables and Vegetable Products",
                },
                {
                    "description": "beets, cooked, boiled",
                    "food_category": "Vegetables and Vegetable Products",
                },
                {
                    "description": "radishes, raw",
                    "food_category": "Vegetables and Vegetable Products",
                },
            ],
            [
                True,
                False,
                False,
            ],  # fresh≈raw, cooked≠raw, different root vegetable species
        ),
        (
            {
                "description": "amaranth grain, uncooked",
                "food_category": "Cereal Grains and Pasta",
            },
            [
                {
                    "description": "amaranth grain, raw",
                    "food_category": "Cereal Grains and Pasta",
                },
                {
                    "description": "amaranth grain, cooked",
                    "food_category": "Cereal Grains and Pasta",
                },
                {
                    "description": "quinoa, uncooked",
                    "food_category": "Cereal Grains and Pasta",
                },
            ],
            [
                True,
                False,
                False,
            ],  # uncooked≈raw, cooked≠uncooked, different grain species
        ),
    ]

    for idx, (ref_desc, cand_list, equiv_list) in enumerate(few_shots, start=1):
        example_payload = [
            {
                "reference": ref_desc,
                "candidates": cand_list,
            }
        ]
        tool_id = f"example_equiv_{idx}"

        # User message with the example
        prompt.append(
            {
                "role": "user",
                "content": (
                    "Determine which candidates are nutritionally equivalent to the reference. JSON list:\n"
                    f"{json.dumps(example_payload, ensure_ascii=False)}"
                ),
            }
        )

        # Assistant message performing the function call
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
                                {"equivalences": [{"is_equivalent": equiv_list}]}
                            ),
                        },
                    }
                ],
            }
        )

        # Matching tool response message (required)
        prompt.append(
            {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": json.dumps(
                    {"equivalences": [{"is_equivalent": equiv_list}]}
                ),
            }
        )

    # ---------------- Actual user batch ----------------
    payload = batch_df.to_dict(orient="records")
    prompt.append(
        {
            "role": "user",
            "content": (
                "For every element in the JSON list, output a list of boolean values indicating whether each candidate is nutritionally equivalent to the reference.\n"
                "Reply ONLY via the function call.\n\n"
                f"JSON list:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
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
        tool_call = response.choices[0].message.tool_calls[0]
        raw_args: str = tool_call.function.arguments  # type: ignore[attr-defined]
    except (AttributeError, IndexError):
        raise RuntimeError("GPT response missing tool-call arguments")

    parsed = BatchModel(**json.loads(raw_args))
    equivalences_df = pd.DataFrame([eq.model_dump() for eq in parsed.equivalences])

    # Preserve index alignment
    equivalences_df.index = batch_df.index
    return equivalences_df


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

    work_df = df.copy()

    # Prepare the data for GPT processing
    work_df["reference"] = work_df[col_reference].apply(str)
    work_df["candidates"] = work_df[col_candidates].apply(
        lambda x: [str(item) for item in x] if isinstance(x, list) else [str(x)]
    )

    # Pre-filter with exact matching for performance and accuracy
    print("[Exact-matching] Pre-filtering obvious matches...")
    exact_matcher = ExactFoodMatcher()

    def _apply_exact_matching(row):
        """Apply exact matching and mark which candidates need GPT processing."""
        reference_desc = _ensure_obj(row["reference"]).get("description", "")
        candidates = row["candidates"]

        candidate_descs = []

        for candidate in candidates:
            candidate_desc = _ensure_obj(candidate).get("description", "")
            candidate_descs.append(candidate_desc)

        # Get exact matches
        exact_results = exact_matcher.find_exact_matches(
            reference_desc, candidate_descs
        )

        # Store exact match results and mark which need GPT
        row["exact_matches"] = exact_results
        row["needs_gpt"] = not all(
            exact_results
        )  # Only send to GPT if not all exact matches

        return row

    work_df = work_df.apply(_apply_exact_matching, axis=1)

    # Filter to only rows that need GPT processing
    gpt_df = work_df[work_df["needs_gpt"]].copy()
    exact_only_df = work_df[~work_df["needs_gpt"]].copy()

    print(
        f"[Exact-matching] {len(exact_only_df)} rows solved by exact matching, {len(gpt_df)} need GPT"
    )

    # Handle exact-only results first
    if len(exact_only_df) > 0:
        exact_only_df["is_equivalent"] = exact_only_df["exact_matches"]

    # Process remaining rows with GPT if any
    if len(gpt_df) > 0:
        # Split GPT rows into batches
        batches = [
            gpt_df.iloc[i : i + batch_size] for i in range(0, len(gpt_df), batch_size)
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

        gpt_results_df = pd.concat(results).sort_index()

        # Merge exact matches with GPT results for partial matches
        def _merge_results(row):
            """Merge exact match results with GPT results where needed."""
            if row.name in gpt_results_df.index:
                gpt_results = gpt_results_df.loc[row.name, "is_equivalent"]
                exact_results = row["exact_matches"]

                # Use exact matches where available, GPT otherwise
                final_results = []
                for i, (exact_match, gpt_result) in enumerate(
                    zip(exact_results, gpt_results)
                ):
                    final_results.append(exact_match if exact_match else gpt_result)

                return final_results
            else:
                return row["exact_matches"]  # Pure exact matches

        gpt_df["is_equivalent"] = gpt_df.apply(_merge_results, axis=1)

        # Combine all results
        equivalence_df = pd.concat([exact_only_df, gpt_df]).sort_index()
    else:
        # Only exact matches
        equivalence_df = exact_only_df

    # Add results to original dataframe
    result_df = df.copy()
    result_df["is_equivalent"] = equivalence_df["is_equivalent"].values

    return result_df
