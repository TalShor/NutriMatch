"""Download and load the "moh_mitzrachim" nutrition database."""

from __future__ import annotations

from typing import List

import pandas as pd
import requests

# Import BaseDataDigestion
from src.base_classes.base_data_digestion import BaseDataDigestion

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

RESOURCE_ID = "c3cb0630-0650-46c1-a068-82d575c094b2"
CSV_URL = (
    "https://data.gov.il/dataset/nutrition-database/resource/"
    f"{RESOURCE_ID}/download/moh_mitzrachim.csv"
)

CKAN_API_ENDPOINT = "https://data.gov.il/api/3/action/datastore_search"
PAGE_SIZE = 50_000  # the API allows up to ~100k; keep it conservative


class ZameretDataDigestion(BaseDataDigestion):
    def __init__(self):
        # Base class expects a data frame parameter that is unused; we pass None.
        super().__init__(fcdb_name="Zameret")

    def download_raw_data(self) -> pd.DataFrame:
        """Fetch the dataset through the CKAN JSON API and convert to DataFrame."""

        # First call – just to learn total number of records
        meta_resp = requests.get(
            CKAN_API_ENDPOINT,
            params={"resource_id": RESOURCE_ID, "limit": 0},
            timeout=30,
        )
        meta_resp.raise_for_status()
        total = meta_resp.json()["result"]["total"]

        print(f"[CKAN] total records = {total}")

        frames: List[pd.DataFrame] = []
        for offset in range(0, total, PAGE_SIZE):
            print(f"  • fetching rows {offset}-{min(offset + PAGE_SIZE, total)} …")
            resp = requests.get(
                CKAN_API_ENDPOINT,
                params={
                    "resource_id": RESOURCE_ID,
                    "limit": PAGE_SIZE,
                    "offset": offset,
                },
                timeout=60,
            )
            resp.raise_for_status()
            records = resp.json()["result"]["records"]
            frames.append(pd.DataFrame.from_records(records))

        return pd.concat(frames, ignore_index=True)

    def standardise_data(self) -> pd.DataFrame:
        sr_legacy_nutrient_translation = {
            "protein": "Protein",
            "total_fat": "Total lipid (fat)",
            "carbohydrates": "Carbohydrate, by difference",
            "food_energy": "Energy",
            "alcohol": "Alcohol, ethyl",
            "moisture": "Water",
            "total_dietary_fiber": "Fiber, total dietary",
            "calcium": "Calcium, Ca",
            "iron": "Iron, Fe",
            "magnesium": "Magnesium, Mg",
            "phosphorus": "Phosphorus, P",
            "potassium": "Potassium, K",
            "sodium": "Sodium, Na",
            "zinc": "Zinc, Zn",
            "copper": "Copper, Cu",
            "vitamin_a_iu": "Vitamin A, IU",
            "carotene": "Carotene, beta",
            "vitamin_e": "Vitamin E (alpha-tocopherol)",
            "vitamin_c": "Vitamin C, total ascorbic acid",
            "thiamin": "Thiamin",
            "riboflavin": "Riboflavin",
            "niacin": "Niacin",
            "vitamin_b6": "Vitamin B-6",
            "folate": "Folate, total",
            "folate_dfe": "Folate, DFE",
            "vitamin_b12": "Vitamin B-12",
            "cholesterol": "Cholesterol",
            "saturated_fat": "Fatty acids, total saturated",
            "butyric": "SFA 4:0",  # assuming butyric acid
            "caproic": "SFA 6:0",  # assuming caproic acid
            "caprylic": "SFA 8:0",  # assuming caprylic acid
            "capric": "SFA 10:0",  # assuming capric acid
            "lauric": "SFA 12:0",  # assuming lauric acid
            "myristic": "SFA 14:0",  # assuming myristic acid
            "palmitic": "SFA 16:0",  # assuming palmitic acid
            "stearic": "SFA 18:0",  # assuming stearic acid
            "oleic": "MUFA 18:1 c",  # assuming oleic acid
            "linoleic": "PUFA 18:2 n-6 c,c",  # assuming linoleic acid
            "linolenic": "PUFA 18:3 n-3 c,c,c (ALA)",  # assuming alpha-linolenic acid
            "arachidonic": "PUFA 20:4 n-6",  # assuming arachidonic acid
            "docosahexanoic": "PUFA 22:6 n-3 (DHA)",  # assuming docosahexaenoic acid
            "palmitoleic": "MUFA 16:1",  # assuming palmitoleic acid
            "parinaric": "PUFA 18:4",  # assuming parinaric acid
            "gadoleic": "MUFA 20:1",  # assuming gadoleic acid
            "eicosapentaenoic": "PUFA 20:5 n-3 (EPA)",  # assuming eicosapentaenoic acid
            "erucic": "MUFA 22:1 c",  # assuming erucic acid
            "docosapentaenoic": "PUFA 22:5 n-3 (DPA)",  # assuming docosapentaenoic acid
            "mono_unsaturated_fat": "Fatty acids, total monounsaturated",
            "poly_unsaturated_fat": "Fatty acids, total polyunsaturated",
            "vitamin_d": "Vitamin D (D2 + D3)",
            "total_sugars": "Sugars, Total",
            "trans_fatty_acids": "Fatty acids, total trans",
            "vitamin_a_re": "Vitamin A, RAE",
            "isoleucine": "Isoleucine",
            "leucine": "Leucine",
            "valine": "Valine",
            "lysine": "Lysine",
            "threonine": "Threonine",
            "methionine": "Methionine",
            "phenylalanine": "Phenylalanine",
            "tryptophan": "Tryptophan",
            "histidine": "Histidine",
            "tyrosine": "Tyrosine",
            "arginine": "Arginine",
            "cystine": "Cystine",
            "serine": "Serine",
            "pantothenic_acid": "Pantothenic acid",
            "selenium": "Selenium, Se",
            "choline": "Choline, total",
            "manganese": "Manganese, Mn",
            "fructose": "Fructose",
        }

        unique_vars = {
            "vitamin_k": "Vitamin K",
            "iodine": "Iodine",
            "sugar_alcohols": "Sugar alcohols",
            "biotin": "Biotin",
        }

        standardised_data = (
            self.raw_data.rename(columns=sr_legacy_nutrient_translation)
            .rename(columns=unique_vars)
            .rename(columns={"shmmitzrach": "hebrew_name"})[
                ["hebrew_name", "english_name"]
                + list(sr_legacy_nutrient_translation.values())
                + list(unique_vars.values())
            ]
        )
        return standardised_data
