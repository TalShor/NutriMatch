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

# ---------------------------------------------------------------------------
# helper – pick the closest match among *n* candidates
# ---------------------------------------------------------------------------

# Concise system prompt for *selection* task (reference vs. candidates)
_SYSTEM_PROMPT_SELECT = {
    "role": "system",
    "content": (
        "You are a certified dietitian specialising in food substitution.\n\n"
        "INPUT (per element):\n"
        "• reference – a food description\n"
        "• candidates – 1…N alternative descriptions\n"
        "OUTPUT: Return ONLY the function-call with "
        '{ "selections": [ { "closest_idx": <int> } … ] }.\n'
        "Index is 1-based; use −1 ONLY when no candidate is sufficiently similar.\n\n"
        "----------------------------------------------------------------------\n"
        "DECISION HIERARCHY (apply top→bottom)\n"
        "0. Textual Match – After normalising trivial wording differences, if a candidate description is identical to the reference, choose it immediately.\n"
        "1. Core Food Identity – Look at the *description* first. Prefer candidates whose description names the SAME primary food (same species/plant, same cut or product form) allowing for synonyms, spelling, singular–plural, and word-order changes.\n"
        "2. Food Group  – Prefer same USDA Food Group (or clearly synonymous group).\n"
        "3. Source      – Prefer same primary biological origin: cow-milk, goat-milk, sheep-milk, soy, almond, oat, legume, cereal grain, etc.\n"
        "4. State       – Prefer same preparation state (raw, boiled, baked, fried, grilled, dried, powdered, canned, frozen, etc.).\n"
        "5. Macro Match – Accept if calories ±15 %, each macro (protein/fat/carbs) ±20 %.\n"
        "6. Ignore      – Herbs, spices, salt, sweeteners, added vitamins/minerals UNLESS they change macros >5 g per 100 g. Fat-difference tolerance: ≤5 g/100 g.\n"
        "7. Threshold   – If a candidate satisfies ≥75 % of rules 1–5 choose it; otherwise return −1.\n\n"
        "----------------------------------------------------------------------\n"
        "SCENARIO-SPECIFIC GUIDELINES (augment rules above)\n"
        "• Plant-based milks vs. dairy milk: treat fortified soy/almond/oat milks "
        "as potential substitutes for low-fat or skim cow milk if macros fit.\n"
        "• Low-sugar or sugar-free desserts: sucrose↔sweetener swap is acceptable "
        "if calories stay within ±15 %.\n"
        "• Gluten-free baked goods: GF bread/crackers may substitute for wheat "
        "variants as long as grain base and macros align (e.g., rice bread vs. "
        "white bread).\n"
        "• Composite dishes: All MAIN ingredients + cooking method must match; "
        "sauces or herbs may differ. E.g., ‘baked salmon with lemon’ ≈ ‘baked "
        "salmon, plain’. Treat ‘ready-to-eat’ vs. ‘raw ingredient’ as different "
        "states.\n"
        "• Fermented dairy: yogurt ↔ kefir ↔ labneh may substitute if fat% and "
        "added sugar comparable.\n"
        "• Cheese fat levels: 5 % vs. 8 % is acceptable; 5 % vs. 30 % is not.\n"
        "• Oils and spreads: Avocado oil ↔ olive oil but NOT ↔ butter.\n"
        "• Fortified juices: presence of added vitamins is ignored if macros match.\n"
        "• Meat form: Prefer SAME muscle-cut or slice over ground/minced or cured when both are available.\n"
        "• Species priority: When primary species differs (banana vs. plantain, beef vs. turkey), species match outweighs cooking-method match.\n"
        "• Desserts: Match primary macronutrient SOURCE—dairy/egg custards should not be substituted with pure-sugar toppings even if calories align.\n\n"
        "----------------------------------------------------------------------\n"
        "ALIAS / SYNONYM CHEAT-SHEET (non-exhaustive; treat as SAME food for "
        "rules 1-3)\n"
        "— Cheeses —\n"
        "  Kashkaval → Gouda/Edam/Cheddar family (semi-hard yellow cheese)\n"
        "  Pecorino, Manchego → hard/semi-hard sheep cheese (Parmesan-like)\n"
        "  Halloumi → brined grilling cheese (close to Feta in salt/fat)\n"
        "  Tzfatit, ‘salty cheese’, ‘white cheese’ → brined fresh cheese (≈ Feta)\n"
        "  Labneh → strained yogurt / cream-cheese style spread\n"
        "  Port de Salut → semi-soft cow cheese (like Muenster)\n"
        "— Milks & Yogurts —\n"
        "  Almond/oat/soy milk → plant milk category (use fat-free or low-fat milk "
        "as proxy when macros match)\n"
        "  ‘Producer milk’ → whole cow milk\n"
        "— Breads & Baked goods —\n"
        "  ‘Yellow bread’, ‘light rye’ → rye/wheat bread family\n"
        "  ‘Ma’amoul’ → date-filled shortbread cookie\n"
        "  ‘Gluten-free flour mix’ → baking flour substitute (rice/corn/potato).\n"
        "— Meats & Fish —\n"
        "  ‘Seabass’, ‘sea bass’ same fish; raw ↔ cooked, adjust state rule.\n"
        "  ‘Mosht’ fish → tilapia/cichlid family.\n"
        "— Condiments & Herbs —\n"
        "  Za’atar → thyme/oregano/sumac blend\n"
        "  Amba → pickled mango sauce\n"
        "  ‘Asian sauce (generic)’ ↔ soy-based stir-fry sauce if macros similar.\n"
        "— Processed & Cured Meats —\n"
        "  Poultry pastrami → pastrami, turkey | chicken\n"
        "— Beef Cuts —\n"
        "  Entrecôte → rib-eye steak, boneless beef rib\n"
        "— Misc —\n"
        "  ‘Yellow cheese’ (without spec) → semi-hard cow cheese\n"
        "  ‘Energy bar’ ↔ cereal/protein bar; compare macros not brand.\n\n"
        "----------------------------------------------------------------------\n"
        "EXPLICIT REMINDER: When in doubt CHOOSE the closest qualifying candidate "
        "rather than returning −1.\n"
    ),
}


class GroupSelection(BaseModel):
    """Structured GPT output: index of the closest candidate (or ‑1)."""

    closest_idx: int = Field(
        ...,
        description=(
            "1-based position of the closest candidate within the provided list; "
            "-1 when none of the candidates meet nutritional-similarity criteria."
        ),
    )


# ---------------------------------------------------------------------------
# GPT helpers (per-batch) – selection task
# ---------------------------------------------------------------------------


def _build_select_prompt(batch_df: pd.DataFrame, tool_name: str) -> List[dict]:
    """Compose chat prompt for *selection* batches."""

    prompt: List[dict] = [_SYSTEM_PROMPT_SELECT]

    # ---------------- Few-shot examples to prime the model -----------------
    few_shots = [
        (
            {
                "description": "boiled potato",
                "food_category": "Vegetables and Vegetable Products",
            },
            [
                {
                    "description": "baked potato",
                    "food_category": "Vegetables and Vegetable Products",
                },
                {
                    "description": "fried potato",
                    "food_category": "Vegetables and Vegetable Products",
                },
                {
                    "description": "steamed broccoli",
                    "food_category": "Vegetables and Vegetable Products",
                },
                {
                    "description": "white rice, cooked",
                    "food_category": "Cereal Grains and Pasta",
                },
                {
                    "description": "roasted sweet potato",
                    "food_category": "Vegetables and Vegetable Products",
                },
            ],
            1,
        ),
        (
            {
                "description": "baked salmon with lemon juice, no added oil",
                "food_category": "Finfish and Shellfish Products",
            },
            [
                {
                    "description": "fish, salmon, baked, dry heat",
                    "food_category": "Finfish and Shellfish Products",
                },
                {
                    "description": "fish, salmon, raw",
                    "food_category": "Finfish and Shellfish Products",
                },
                {
                    "description": "fish, salmon, smoked",
                    "food_category": "Finfish and Shellfish Products",
                },
                {
                    "description": "fish, trout, baked",
                    "food_category": "Finfish and Shellfish Products",
                },
                {
                    "description": "fish, seabass, baked",
                    "food_category": "Finfish and Shellfish Products",
                },
            ],
            1,
        ),
        (
            {
                "description": "green peas, raw",
                "food_category": "Vegetables and Vegetable Products",
            },
            [
                {
                    "description": "green peas, frozen, uncooked",
                    "food_category": "Vegetables and Vegetable Products",
                },
                {
                    "description": "green beans, raw",
                    "food_category": "Vegetables and Vegetable Products",
                },
                {
                    "description": "black beans, cooked",
                    "food_category": "Legumes and Legume Products",
                },
                {
                    "description": "spinach, raw",
                    "food_category": "Vegetables and Vegetable Products",
                },
                {
                    "description": "green peas, canned, drained",
                    "food_category": "Vegetables and Vegetable Products",
                },
            ],
            -1,
        ),
        (
            {
                "description": "almonds, roasted, salted",
                "food_category": "Nut and Seed Products",
            },
            [
                {
                    "description": "almonds, raw",
                    "food_category": "Nut and Seed Products",
                },
                {
                    "description": "cashews, roasted, salted",
                    "food_category": "Nut and Seed Products",
                },
                {
                    "description": "peanuts, roasted, salted",
                    "food_category": "Legumes and Legume Products",
                },
                {
                    "description": "almond butter, plain",
                    "food_category": "Nut and Seed Products",
                },
                {
                    "description": "walnuts, raw",
                    "food_category": "Nut and Seed Products",
                },
            ],
            1,
        ),
        (
            {
                "description": "baked canned baked beans in tomato sauce",
                "food_category": "Legumes and Legume Products",
            },
            [
                {
                    "description": "beans, baked, canned, with pork and tomato sauce",
                    "food_category": "Legumes and Legume Products",
                },
                {
                    "description": "beans, baked, canned, with beef",
                    "food_category": "Legumes and Legume Products",
                },
                {
                    "description": "beans, baked, canned, with pork",
                    "food_category": "Legumes and Legume Products",
                },
                {
                    "description": "beans, baked, canned, with franks",
                    "food_category": "Legumes and Legume Products",
                },
                {
                    "description": "beans, baked, canned, plain or vegetarian",
                    "food_category": "Legumes and Legume Products",
                },
            ],
            5,
        ),
        (
            {
                "description": "arabian dough stuffed with za'atar herb (homemade)",
                "food_category": "Baked Products",
            },
            [
                {"description": "phyllo dough", "food_category": "Baked Products"},
                {"description": "bread, cinnamon", "food_category": "Baked Products"},
                {
                    "description": "bread, cracked wheat",
                    "food_category": "Baked Products",
                },
                {
                    "description": "bread, pita, whole wheat",
                    "food_category": "Baked Products",
                },
                {"description": "bread, wheat", "food_category": "Baked Products"},
            ],
            4,
        ),
        (
            {
                "description": "apple, raw, with skin",
                "food_category": "Fruits and Fruit Juices",
            },
            [
                {
                    "description": "apples, raw, without skin",
                    "food_category": "Fruits and Fruit Juices",
                },
                {
                    "description": "apples, raw, golden delicious, with skin",
                    "food_category": "Fruits and Fruit Juices",
                },
                {
                    "description": "apples, raw, with skin (includes foods for usda's food distribution program)",
                    "food_category": "Fruits and Fruit Juices",
                },
                {
                    "description": "apples, raw, without skin, cooked, boiled",
                    "food_category": "Fruits and Fruit Juices",
                },
                {
                    "description": "apples, raw, without skin, cooked, microwave",
                    "food_category": "Fruits and Fruit Juices",
                },
            ],
            3,
        ),
        (
            {
                "description": "baked falafel (commercial)",
                "food_category": "Fast Foods",
            },
            [
                {
                    "description": "falafel, home prepared",
                    "food_category": "Legumes and Legume Products",
                },
                {
                    "description": "hummus, commercial",
                    "food_category": "Legumes and Legume Products",
                },
                {
                    "description": "smart soup, moroccan chick pea",
                    "food_category": "Meals, Entrees, and Side Dishes",
                },
                {
                    "description": "hummus, home prepared",
                    "food_category": "Legumes and Legume Products",
                },
                {
                    "description": "cake, cheesecake, commercially prepared",
                    "food_category": "Baked Products",
                },
            ],
            1,
        ),
        (
            {
                "description": "bulgarian cheese (cow's milk), 5 % fat, sliced",
                "food_category": "Dairy and Egg Products",
            },
            [
                {
                    "description": "cheese, cream, low fat",
                    "food_category": "Dairy and Egg Products",
                },
                {
                    "description": "cheese, feta",
                    "food_category": "Dairy and Egg Products",
                },
                {
                    "description": "cheese, swiss, low fat",
                    "food_category": "Dairy and Egg Products",
                },
                {
                    "description": "cheese, cream, fat free",
                    "food_category": "Dairy and Egg Products",
                },
                {
                    "description": "cheese, cream",
                    "food_category": "Dairy and Egg Products",
                },
            ],
            2,
        ),
        (
            {
                "description": "bagel (white, plain, sliced)",
                "food_category": "Baked Products",
            },
            [
                {
                    "description": "bagels, whole grain white",
                    "food_category": "Baked Products",
                },
                {"description": "bagels, wheat", "food_category": "Baked Products"},
                {"description": "bagels, egg", "food_category": "Baked Products"},
                {
                    "description": "bagels, plain, unenriched, without calcium propionate (includes onion, poppy, sesame)",
                    "food_category": "Baked Products",
                },
                {
                    "description": "bread, white wheat",
                    "food_category": "Baked Products",
                },
            ],
            2,
        ),
        (
            {
                "description": "baked sea bass (raw or cooked state not specified)",
                "food_category": "Finfish and Shellfish Products",
            },
            [
                {
                    "description": "fish, sea bass, mixed species, raw",
                    "food_category": "Finfish and Shellfish Products",
                },
                {
                    "description": "fish, sea bass, mixed species, cooked, dry heat",
                    "food_category": "Finfish and Shellfish Products",
                },
                {
                    "description": "fish, bass, striped, raw",
                    "food_category": "Finfish and Shellfish Products",
                },
                {
                    "description": "fish, bass, fresh water, mixed species, raw",
                    "food_category": "Finfish and Shellfish Products",
                },
                {
                    "description": "fish, bluefish, raw",
                    "food_category": "Finfish and Shellfish Products",
                },
            ],
            2,
        ),
        (
            {
                "description": "avocado spread (guacamole)",
                "food_category": "Fats and Oils",
            },
            [
                {"description": "oil, avocado", "food_category": "Fats and Oils"},
                {
                    "description": "vegetable oil butter spread, reduced calorie",
                    "food_category": "Fats and Oils",
                },
                {
                    "description": "margarine like, vegetable oil spread, fat free, tub",
                    "food_category": "Fats and Oils",
                },
                {
                    "description": "margarine like, vegetable oil spread, 20% fat, without salt",
                    "food_category": "Fats and Oils",
                },
                {
                    "description": "margarine like, vegetable oil spread, 20% fat, with salt",
                    "food_category": "Fats and Oils",
                },
            ],
            -1,
        ),
        (
            {"description": "balsamic vinegar", "food_category": "Spices and Herbs"},
            [
                {
                    "description": "vinegar, balsamic",
                    "food_category": "Spices and Herbs",
                },
                {
                    "description": "vinegar, distilled",
                    "food_category": "Spices and Herbs",
                },
                {
                    "description": "vinegar, red wine",
                    "food_category": "Spices and Herbs",
                },
                {"description": "vinegar, cider", "food_category": "Spices and Herbs"},
                {
                    "description": "spices, basil, dried",
                    "food_category": "Spices and Herbs",
                },
            ],
            1,
        ),
        # Example demonstrating Rule 0 – textual equivalence (banana vs. bananas)
        (
            {
                "description": "banana, raw",
                "food_category": "Fruits and Fruit Juices",
            },
            [
                {
                    "description": "bananas, raw",
                    "food_category": "Fruits and Fruit Juices",
                },
                {
                    "description": "plantains, yellow, raw",
                    "food_category": "Fruits and Fruit Juices",
                },
                {
                    "description": "papayas, raw",
                    "food_category": "Fruits and Fruit Juices",
                },
                {
                    "description": "breadfruit, raw",
                    "food_category": "Fruits and Fruit Juices",
                },
                {
                    "description": "apricots, raw",
                    "food_category": "Fruits and Fruit Juices",
                },
            ],
            1,
        ),
    ]

    for idx, (ref_desc, cand_list, chosen_idx) in enumerate(few_shots, start=1):
        example_payload = [
            {
                "reference": ref_desc,
                "candidates": cand_list,
            }
        ]
        tool_id = f"example_sel_{idx}"

        # User message with the example
        prompt.append(
            {
                "role": "user",
                "content": (
                    "Choose the nutritionally closest candidate. JSON list:\n"
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
                                {"selections": [{"closest_idx": chosen_idx}]}
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
                "content": json.dumps({"selections": [{"closest_idx": chosen_idx}]}),
            }
        )

    # ---------------- Actual user batch ----------------
    payload = batch_df.to_dict(orient="records")
    prompt.append(
        {
            "role": "user",
            "content": (
                "For every element in the JSON list, output the index (1-based) of the nutritionally closest candidate – or ‑1 if none qualify.\n"
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
def _query_select_gpt(batch_df: pd.DataFrame) -> pd.DataFrame:
    """Send one *selection* batch to GPT and parse the response."""

    batch_size = len(batch_df)

    # Dynamic model enforcing exact list length
    BatchModel = create_model(
        "BatchGroupSelections",
        selections=(
            conlist(GroupSelection, min_length=batch_size, max_length=batch_size),
            ...,
        ),
    )

    tool_spec = openai.pydantic_function_tool(BatchModel)
    tool_name = tool_spec["function"]["name"]

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=_build_select_prompt(batch_df, tool_name),
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
    selections_df = pd.DataFrame([sel.model_dump() for sel in parsed.selections])

    # Preserve index alignment
    selections_df.index = batch_df.index
    return selections_df


# ---------------------------------------------------------------------------
# Public helper – closest-candidate selection
# ---------------------------------------------------------------------------


def select_closest_dataframe(
    df: pd.DataFrame,
    col_item_a: str | None = None,
    col_item_b: str | None = None,
    n_candidates: int = 5,
    batch_size: int = 30,
    num_threads: int = 8,
) -> pd.DataFrame:
    """For each reference item (column *A*), choose the closest of *n* candidates.

    The dataframe is expected to contain exactly *n_candidates* rows per unique
    value in *col_item_a* (the reference item).  Column *col_item_b* holds a
    **different** candidate in every row.  The function asks GPT to decide
    which candidate is nutritionally most similar and returns a *reduced*
    dataframe with one row per reference item and an integer column
    ``closest_idx`` holding 1…*n_candidates* (or ‑1 when none qualify).
    """

    if col_item_a is None or col_item_b is None:
        col_item_a, col_item_b = df.columns[:2]

    # Helper to ensure dictionary objects (not JSON strings)
    def _ensure_obj(val):
        return val if not isinstance(val, str) else json.loads(val)

    # Extract *description* strings only (fallback to str(obj) if missing)
    # def _desc(obj):
    #     if isinstance(obj, dict) and "description" in obj:
    #         return obj["description"]
    #     return str(obj)

    work_df = df[[col_item_a, col_item_b]].copy()
    work_df[col_item_a] = work_df[col_item_a].apply(str)
    work_df[col_item_b] = work_df[col_item_b].apply(str)

    # Basic sanity checks ----------------------------------------------------
    counts = work_df.groupby(col_item_a).size()
    if (counts != n_candidates).any():
        raise ValueError(
            "Each reference item must have exactly n_candidates candidates."
        )

    # Build grouped dataframe with one row per reference item ---------------
    groups = []
    references = []  # keep original reference objects for final output

    grouped_df = work_df.groupby(col_item_a)[col_item_b].apply(list).reset_index()
    grouped_df.columns = ["reference", "candidates"]

    # Split into batches -----------------------------------------------------
    batches = [
        grouped_df.iloc[i : i + batch_size]
        for i in range(0, len(grouped_df), batch_size)
    ]
    if not batches:
        return pd.DataFrame(columns=[col_item_a, "closest_idx"])

    # Parallel GPT execution -------------------------------------------------
    results: List[pd.DataFrame] = []

    def _run(batch: pd.DataFrame):
        print(f"[GPT-select] reference rows {batch.index.min()}–{batch.index.max()}")
        return _query_select_gpt(batch)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_map = {
            executor.submit(_run, batch): idx for idx, batch in enumerate(batches)
        }
        for fut in concurrent.futures.as_completed(future_map):
            try:
                results.append(fut.result())
            except Exception as exc:
                idx = future_map[fut]
                raise RuntimeError(f"Failed GPT selection for batch {idx}") from exc

    selection_df = pd.concat(results).sort_index()

    # Build reduced output dataframe ----------------------------------------
    matches_ranks = pd.DataFrame(
        {
            "reference": grouped_df["reference"],
            "closest_idx": selection_df["closest_idx"].values,
        }
    )

    df_out = df.copy()
    df_out["reference"] = df_out[col_item_a].apply(str)

    df_out = (
        df_out.groupby("reference")[col_item_b]
        .apply(list)
        .to_frame(f"{col_item_b}_options")
        .join(matches_ranks.set_index("reference"))
    )
    df_out.index = pd.Series(df_out.index).apply(literal_eval)
    return df_out
