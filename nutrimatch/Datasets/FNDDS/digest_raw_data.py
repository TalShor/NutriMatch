import numpy as np
import pandas as pd
from ...Utils import Paths


def get_nutrients():
    fndds_paths = Paths('FNDDS')
    nutrient_levels = pd.read_json(fndds_paths.raw_data_path + '/FoodData_Central_survey_food_json_2022-10-28.json', orient='records')
    nutrient_levels = pd.json_normalize(nutrient_levels['SurveyFoods'])
    nutrient_values = nutrient_levels.set_index('description')\
        ['foodNutrients'].explode()\
        .apply(pd.Series)\
        .assign(nutrient_name = lambda df: df.nutrient.apply(lambda x: x['name']))

    nutrient_values\
        .pivot_table(index='description', columns='nutrient_name', values='amount')\
        .to_parquet(fndds_paths.nutrients_path)
    
# extracting food portions mapping

# def get_food_portions():
#     fndds_paths = Paths('FNDDS')
#     df  = pd.read_parquet(Paths('FNDDS').food_items_path)
#     foodPortions = df.set_index('description')\
#         ['foodPortions'].explode()\
#         .apply(pd.Series)
#     foodPortions.to_parquet(fndds_paths.food_portions_path)


if __name__ == "__main__":
    get_nutrients()