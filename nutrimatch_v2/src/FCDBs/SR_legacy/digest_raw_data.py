import io
import zipfile

import pandas as pd
import requests

from ...base_classes.base_data_digestion import BaseDataDigestion


class SR_LegacyDataDigestion(BaseDataDigestion):
    def __init__(self):
        # The base class handles downloading and saving.  Pass the FCDB name.
        super().__init__(fcdb_name="SR_legacy")

    def download_raw_data(self) -> pd.DataFrame:
        url = (
            "https://fdc.nal.usda.gov/fdc-datasets/"
            "FoodData_Central_sr_legacy_food_json_2018-04.zip"
        )

        # 1. Download the zip to memory
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()

        # 2. Open the zipfile from the bytes
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            # the archive contains a single JSON file:
            json_name = [n for n in z.namelist() if n.endswith(".json")][0]
            print("found:", json_name)

            # 3. read it directly into pandas
            with z.open(json_name) as f:
                df = pd.read_json(f)["SRLegacyFoods"].apply(pd.Series)
                df["foodCategory"] = df.foodCategory.apply(pd.Series)["description"]
                return df

    def standardise_data(self) -> pd.DataFrame:
        nutrients_table = (
            self.raw_data.set_index(["description", "foodCategory"])["foodNutrients"]
            .explode()
            .apply(pd.Series)
        )
        nutrients_table = nutrients_table[["amount"]].join(
            nutrients_table["nutrient"].apply(pd.Series)[["name"]]
        )

        return nutrients_table.reset_index().pivot_table(
            index=["description", "foodCategory"], values="amount", columns="name"
        )
