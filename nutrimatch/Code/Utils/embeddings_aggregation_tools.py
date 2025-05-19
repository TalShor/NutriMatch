from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

from .paths import Paths


def get_embedding(
        main_dataset_embedding_name="SR_Legacy", 
        dataset_names=["Zameret", "HPP", "FNDDS", "SR_Legacy"]):
    """
    Get the embeddings for the given dataset names.
    The embeddings are returned as a pandas DataFrame with the embedding as the only column and the dataset and food_name as the index.

    Args:
        main_dataset_embedding_name (str): The name of the main dataset embedding.
        dataset_names (list): The names of the datasets to get embeddings for.

    Returns:
        pd.DataFrame: The embeddings as a pandas DataFrame with the embedding as the only column and the dataset and food_name as the index.
    """

    col_renaming_dict = {
        "SR_Legacy": {"SR_Legacy_embedding":"embedding", "description":"food_name"},
        "Zameret": {"Zameret_to_SR_Legacy_embedding":"embedding", "hebrew_name":"food_name"},
        "HPP": {"HPP_to_SR_Legacy_embedding":"embedding", "hebrew_name":"food_name"},
        "FNDDS": {"FNDDS_to_SR_Legacy_embedding":"embedding", "description":"food_name"},
        "Aus": {"Aus_to_SR_Legacy_embedding":"embedding", "food_name":"food_name"}
    }
    engines = {
        "SR_Legacy": "pyarrow",
        "Zameret": "fastparquet",
        "HPP": "pyarrow",
        "FNDDS": "pyarrow",
        "Aus": "pyarrow"
    }
    embeddings = []
    for dataset_name in dataset_names:
        embedding = pd.read_parquet(
            Paths(main_dataset_embedding_name, dataset_name).embedding_path,
            engine=engines[dataset_name])\
            .assign(dataset=dataset_name)
        
        # rename columns if needed
        if dataset_name in col_renaming_dict:
            embedding = embedding.rename(columns=col_renaming_dict[dataset_name])
        
        # filter for final phase
        if "phase" in embedding.columns:
            embedding = embedding.query('phase == "final"')

        embeddings.append(
            embedding.set_index(['dataset', 'food_name'])[['embedding']])

    all_embeddings = pd.concat(embeddings)
    return all_embeddings


def get_distances(all_embeddings):
    """
    Get the distances between the embeddings of the food items in the dataset.
    """
    distances = pd.DataFrame(
        cosine_similarity(
            np.stack(all_embeddings.embedding)), 
        index=all_embeddings.index, 
        columns=all_embeddings.index)
    return distances


if __name__ == "__main__":
    all_embeddings = get_embedding('SR_Legacy', ['HPP', 'FNDDS', 'Zameret', 'SR_Legacy'])
    all_embeddings.to_parquet('s3://datasets-development/diet/food_registry/HPP/nutrients_from_zameret_fndds_sr_legacy/all_embeddings.parquet')

    get_distances(all_embeddings)\
        .to_parquet('s3://datasets-development/diet/food_registry/HPP/nutrients_from_zameret_fndds_sr_legacy/distances.parquet')