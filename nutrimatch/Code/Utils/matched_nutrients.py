import pandas as pd
from .paths import Paths
import numpy as np

def get_matches_in_format(matched_path, dataset_name="HPP"):
    """
    Get the matches XXX
    """
    matches = pd.read_parquet(matched_path, engine='fastparquet')

    # change the index to food_name
    matches = matches\
        .reset_index('item').rename(columns={'item':'food_name'}).set_index('food_name', append=True)
    
    # filter for the dataset
    matches = matches.loc[dataset_name]
    
    # melt the matches to get the matched food names and scores
    all_matches = matches\
        .melt(ignore_index=False, var_name='matched_dataset', value_name='matched_food_name')\
        .query('matched_food_name.str.len() > 0')\
        .set_index('matched_dataset', append=True)\
        .explode(column='matched_food_name')\
        .matched_food_name.apply(pd.Series)
    
    # rename the columns to matched_food_name and matched_food_score
    cleaned_matches = all_matches\
        .rename(columns={0:'matched_food_name', 1:'matched_food_score'})\
        .sort_values(by='matched_food_score', ascending=False)

    return cleaned_matches


def get_nutrients(dataset_names = ['Zameret', 'SR_Legacy', 'FNDDS']):
    """
    Get the nutrients for the given dataset names.

    Args:
        dataset_names (list): The names of the datasets to get nutrients for.

    Returns:
        pd.DataFrame: The nutrients as a pandas DataFrame with the nutrients as the only column and the dataset and food_name as the index.
    """
    nutrients = []
    for dataset_name in dataset_names:
        nutrients.append(pd.read_parquet(Paths(dataset_name).nutrients_path)\
            .assign(matched_dataset=dataset_name))
    
    nutrients = pd.concat(nutrients)
    nutrients.index.name = 'matched_food_name'
    nutrients.set_index('matched_dataset', append=True, inplace=True)
    return nutrients


def matched_nutrients(
    matches:pd.DataFrame, nutrients:pd.DataFrame, 
    dataset_of_interest:str = 'HPP', dataset_names:list = ['FNDDS', 'SR_Legacy', 'Zameret']):
    """
    For each of the matched food items to our food items - take the nutrients from the matched food items

    Args:
        matches (pd.DataFrame): The matches between our food items and the external datasets.
        nutrients (pd.DataFrame): The nutrients for the external datasets.
        dataset_of_interest (str): The dataset of interest.
        dataset_names (list): The names of the external datasets.

    Returns:
        pd.DataFrame: The nutrients for the matched food items.
    """
    matched_nutrients_list = []
    for dataset in ['FNDDS', 'SR_Legacy', 'Zameret']:
        matched_nutrients_list.append(
            matches.loc[(slice(None), dataset), :]\
                .reset_index(['food_name'])\
                .set_index('matched_food_name', append=True)\
            .join(nutrients)\
            .reset_index('matched_dataset').reset_index(drop=True)
        )
    matched_nutrients = pd.concat(matched_nutrients_list)

    # low priority datasets are Zameret and HPP
    matched_nutrients['ranking'] = matched_nutrients['matched_dataset']\
        .apply(lambda x: 1 if x == 'Zameret' or x == 'HPP' else 2)
    
    # replace FNDDS with SR_Legacy (since they are the same from the perspective of the external datasets)
    matched_nutrients['matched_dataset'] = matched_nutrients['matched_dataset']\
        .str.replace('FNDDS', 'SR_Legacy')
    
    # if a matched food name has multiple matches, take the mean of the nutrients
    matched_nutrients = matched_nutrients\
        .sort_values(by='matched_food_score', ascending=False)\
        .groupby(['food_name', 'ranking']).mean(1) # Used to be head(1)

    matched_nutrients = matched_nutrients.sort_values(by='ranking', ascending=False)

    # if a matched food name has multiple matches, take the first non-null nutrient
    def first_non_null(series):
        idx = series.first_valid_index()
        return series.at[idx] if idx is not None else np.nan
    
    grouped_nutrients = matched_nutrients\
        .groupby(['food_name'])\
        .agg(first_non_null)
    
    return grouped_nutrients


def full_nutrients_table(grouped_nutrients, nutrients, all_embeddings, dataset_of_interest:str = 'HPP'):
    """
    Combine the matched nutrients with the nutrients from the external datasets.
    """
    nutrients_fixed = pd.concat([
        grouped_nutrients.reset_index().assign(dataset=dataset_of_interest), 
        nutrients.reset_index()\
            .rename(columns={'matched_food_name':'food_name', 'matched_dataset':'dataset'}), 
    ]).set_index(['dataset', 'food_name'])
    return nutrients_fixed.reindex(all_embeddings.index)


if __name__ == "__main__":
    matches = get_matches_in_format('s3://datasets-development/diet/dataset_alignment/AggregatedMap/top_5_HPP_SR_Legacy_Zameret_FNDDS.parquet')
    matches.to_parquet('s3://datasets-development/diet/food_registry/HPP/nutrients_from_zameret_fndds_sr_legacy/matches.parquet')

    nutrients = get_nutrients(['Zameret', 'SR_Legacy', 'FNDDS'])
    nutrients.to_parquet('s3://datasets-development/diet/food_registry/HPP/nutrients_from_zameret_fndds_sr_legacy/nutrients.parquet')

    grouped_nutrients = matched_nutrients(matches, nutrients)
    grouped_nutrients.to_parquet('s3://datasets-development/diet/food_registry/HPP/nutrients_from_zameret_fndds_sr_legacy/grouped_nutrients.parquet')

    all_embeddings = pd.read_parquet('s3://datasets-development/diet/food_registry/HPP/nutrients_from_zameret_fndds_sr_legacy/all_embeddings.parquet')
    full_nutrients_table(grouped_nutrients, nutrients, all_embeddings, dataset_of_interest='HPP')\
        .to_parquet('s3://datasets-development/diet/food_registry/HPP/nutrients_from_zameret_fndds_sr_legacy/full_nutrients_table.parquet')

