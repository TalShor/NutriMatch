import pandas as pd
import os
from typing import List
from ...Utils.paths import Paths


def sr_legacy(main_dir:str) -> dict:
    """
    This function reads the food and food_category tables from the sr_legacy database and returns a single dataframe with the following columns:
    - description: the name of the food
    - food_category: the category of the food
    - fdc_id: the id of the food in the USDA database
    
    The sr_legacy database is located in the 'sr_legacy' folder in the main_dir directory. The food and food_category tables are located in the 'food.csv' and 'food_category.csv' files, respectively. The food_category table is joined with the food table using the 'food_category_id' column.
    'food.csv' [fdc_id	data_type	description	food_category_id	publication_date]
    'food_category.csv' [	id	code	description]
    'food_attribute.csv' [id	fdc_id	seq_num	food_attribute_type_id	name	value]

    Args:
        main_dir (str): the directory where the sr_legacy database is located

    Returns:
        dict: {'food': a dataframe with the following columns: description, food_category, fdc_id

    """

    food = pd.read_csv(os.path.join(main_dir, 'sr_legacy', 'food.csv'))\
        .drop(columns=['data_type', 'publication_date'])
    
    food_category = pd.read_csv(os.path.join(main_dir, 'sr_legacy', 'food_category.csv'))\
        .rename(columns={'id': 'food_category_id', 'description': 'food_category'})\
        .drop(columns=['code'])

    if food['description'].unique().shape[0] != food.shape[0]:
        raise ValueError('Description is not unique')
    
    if food_category['food_category_id'].unique().shape[0] != food_category.shape[0]:
        raise ValueError('food_category_id is not unique')


    # in sr_legacy there is only one food_attribute_type - Common Name (1000) + they are unique to a fdc_id
    food_attribute = pd.read_csv(os.path.join(main_dir, 'sr_legacy', 'food_attribute.csv'))\
        .rename(columns={'value': 'common_name'})\
        .query('food_attribute_type_id == 1000')\
        [['fdc_id', 'common_name']]
    
    food = food.merge(food_category, on='food_category_id', how='left')\
        .drop(columns=['food_category_id'])\
        .merge(food_attribute, on='fdc_id', how='left')\
        
    food = food.reset_index()\
        .set_index('description')

    food_nutrient = pd.read_csv(os.path.join(main_dir, 'sr_legacy', 'food_nutrient.csv'))\
        [['fdc_id', 'nutrient_id', 'amount']].set_index('nutrient_id')
    
    nutrients = pd.read_csv(os.path.join(main_dir, 'sr_legacy', 'nutrient.csv'))\
        [['id', 'name', 'unit_name']].set_index('id')
    
    # apart from energy - they all have a single unit_name
    nutrients_mapped = food_nutrient.join(nutrients, how='left')\
        .pivot_table(index='fdc_id', columns=['name', 'unit_name'], values='amount', aggfunc='first')

    # convert the Energy, kJ to kcal
    only_kJ = nutrients_mapped[('Energy', 'KCAL')].isnull()
    nutrients_mapped.loc[only_kJ, ('Energy', 'KCAL')] = nutrients_mapped.loc[only_kJ, ('Energy', 'kJ')] / 4.184

    nutrients_mapped.drop(columns=[('Energy', 'kJ')], inplace=True)
    measurements = pd.DataFrame({'unit':nutrients_mapped.columns.get_level_values(1)}, index=nutrients_mapped.columns.get_level_values(0))
    # take only the first level of the multiindex
    nutrients_mapped.columns = nutrients_mapped.columns.get_level_values(0)
    nutrients_mapped = nutrients_mapped.join(food[['fdc_id']].reset_index().set_index('fdc_id'))\
        .set_index('description')
    return {'food': food, 'nutrients_mapped': nutrients_mapped, 'measurements': measurements}



if __name__ == "__main__":
    main_dir = 's3://datasets-development/diet/USDA/raw'
    tables = sr_legacy(main_dir)
    tables['food'].to_parquet('s3://datasets-development/diet/USDA/text/sr_foods.parquet')
    tables['nutrients_mapped'].to_parquet('s3://datasets-development/diet/USDA/text/sr_nutrients.parquet')
    tables['measurements'].to_csv('s3://datasets-development/diet/USDA/text/sr_nutrients_units.csv')


# def food_porition_file():
#     'function for creating food portion file to be saved in (Paths('SR_Legacy').food_portions_path)'
#     sr_legacy_food_item =  pd.read_parquet(Paths('SR_Legacy').food_items_path)
#     food_portion = pd.read_csv('s3://datasets-development/diet/dataset_alignment/SR_Legacy/RawData/food_portion.csv')
#     sr_legacy_food_portions = sr_legacy_food_portions.merge(sr_legacy_food_item[['fdc_id', 'description']], on='fdc_id', how='left')