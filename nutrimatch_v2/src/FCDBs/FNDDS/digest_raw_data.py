import io

import pandas as pd
import requests

from ...base_classes.base_data_digestion import BaseDataDigestion

# category breakdown: https://www.ars.usda.gov/ARSUserFiles/80400530/pdf/2123/Food_Category_List_2021-2023.pdf


class FNDDSDataDigestion(BaseDataDigestion):
    def __init__(self, *args, **kwargs):
        # The base class handles downloading and saving.  Pass the FCDB name.
        super().__init__(fcdb_name="FNDDS", *args, **kwargs)

    #
    def download_raw_data(self) -> pd.DataFrame:
        # USDA "FNDDS Nutrient Values" dataset is provided as an Excel workbook that
        # can be downloaded directly via HTTPS.  We pull the file into memory and
        # hand it off to `pandas.read_excel`.

        url = (
            "https://www.ars.usda.gov/ARSUserFiles/80400530/apps/"
            "2021-2023%20FNDDS%20At%20A%20Glance%20-%20FNDDS%20Nutrient%20Values.xlsx"
        )

        # 1. Download the XLSX file to memory
        headers = {
            # Pretend to be a regular browser to bypass basic bot protections
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "*/*",
            "Referer": "https://www.ars.usda.gov/",
        }

        resp = requests.get(url, headers=headers, timeout=120)
        resp.raise_for_status()

        # 2. Read the Excel workbook directly from the downloaded bytes
        # The first sheet (index 0) contains the nutrient data.
        # If the structure changes in future releases, adjust `sheet_name` accordingly.
        df = pd.read_excel(io.BytesIO(resp.content), sheet_name=0, skiprows=1)  # type: ignore

        # TODO: Map/rename columns to the expected schema if required by downstream
        #       processing.  For now we return the raw dataframe as-is.

        return df


# import numpy as np
# import pandas as pd
# from ...Utils import Paths


# def get_nutrients():
#     fndds_paths = Paths('FNDDS')
#     nutrient_levels = pd.read_json(fndds_paths.raw_data_path + '/FoodData_Central_survey_food_json_2022-10-28.json', orient='records')
#     nutrient_levels = pd.json_normalize(nutrient_levels['SurveyFoods'])
#     nutrient_values = nutrient_levels.set_index('description')\
#         ['foodNutrients'].explode()\
#         .apply(pd.Series)\
#         .assign(nutrient_name = lambda df: df.nutrient.apply(lambda x: x['name']))

#     nutrient_values\
#         .pivot_table(index='description', columns='nutrient_name', values='amount')\
#         .to_parquet(fndds_paths.nutrients_path)

# # extracting food portions mapping

# # def get_food_portions():
# #     fndds_paths = Paths('FNDDS')
# #     df  = pd.read_parquet(Paths('FNDDS').food_items_path)
# #     foodPortions = df.set_index('description')\
# #         ['foodPortions'].explode()\
# #         .apply(pd.Series)
# #     foodPortions.to_parquet(fndds_paths.food_portions_path)


# if __name__ == "__main__":
#     get_nutrients()
