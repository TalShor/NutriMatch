import argparse
import pandas as pd
import sys
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pydantic import BaseModel, Field
from typing import Any
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import heapq
import dask.dataframe as dd
sys.path.append('fdc_gpt_alignment/')
from Code import Paths, SR_LegacyFoodItem, HPPFoodItem, HPPFoodItem, NearEastFoodItem, MEXT2015FoodItem, ZameretFoodItem, FooDBFoodItem, FNDDSFoodItem
from Code.Utils.production import check_s3_path_exists

"""
Food Registry - Cross Dataset Mapping

This script performs cross-dataset mapping for food items across multiple datasets.
All similarities are compared in the same embedding space (e.g., SR_Legacy space).

Process:
1. Load pre-computed embeddings for all required datasets in the specified space.
2. Concatenate embeddings from all datasets into a single DataFrame.
3. Compute cosine similarity between all embeddings.
4. Generate a distance matrix and identify top N matches for each food item across datasets.
5. Save the top matches for each dataset combination in parquet files.

Key Features:
- Supports multiple datasets and embedding spaces.
- Configurable number of top matches to retrieve.
- Efficient similarity computation using optimized libraries.
- Parallel processing for improved performance on large datasets.
- Results saved in easily accessible parquet format for further analysis.

Usage:
Run this script with appropriate command-line arguments to specify datasets,
embedding space, and number of top matches to retrieve.
"""

def get_top_n_matches_for_item(item_index, distances, top_n: int, target_dataset: str):
    """
    Retrieve the top n matches for a single item from the target dataset.
    Args:
        item_index (tuple): Index of the item in the distance matrix.
        distances (pd.DataFrame): Distance matrix.
        top_n (int): Number of top matches to retrieve.
        target_dataset (str): Name of the target dataset.
        column_names (dict[str, dict[Any, Any]]): Dictionary containing the column names for each dataset.
    """
    item_similarities = distances.loc[item_index].drop(item_index)
    
    # Filter similarities to only those from the target dataset
    
    dataset_similarities = item_similarities[item_similarities.index.get_level_values(0) == target_dataset]

    top_matches = heapq.nlargest(top_n, dataset_similarities.items(), key=lambda x: x[1])

    return top_matches

def cross_dataset_mapping(datasets: list[str], embedding_space: str, top_n_matches: int, overwrite: bool = False) -> None:
    """
    Perform cross-dataset mapping by calculating cosine similarity between embeddings.
    Args:
        datasets (list[str]): List of dataset names.
        embedding_space (str): Name of the embedding space.
        top_n_matches (int): Number of top matches to retrieve for each item in each dataset.        
        overwrite (bool, optional): Whether to overwrite existing top matches. Defaults to False.
    """
    

    # Get embeddings in the given space for all the required datasets.
    embeddings = []
    column_names = {}
    embedding_space_item = globals()[f"{embedding_space}FoodItem"]
    embedding_space_unique_food_column = f"{embedding_space}__{embedding_space_item.unique_food_column}"
    for dataset in datasets:
        path = Paths(embedding_space, dataset).embedding_path        
        column_names[dataset] = {
            "unique_food_column": globals()[f"{dataset}FoodItem"].unique_food_column
        }
        try:
            df = dd.read_parquet(path).compute()
        except:
            print(f"Embedding file not found for {dataset} in {embedding_space}")
            return
        df["dataset"] = dataset
        embedding_column_name = f"{dataset}_to_{embedding_space}_embedding"
        if dataset == embedding_space:
            df[embedding_space_unique_food_column] = df[embedding_space_item.unique_food_column]
            embedding_column_name = f"{dataset}_embedding"
        df.set_index(["dataset", embedding_space_unique_food_column, column_names[dataset]["unique_food_column"]], inplace=True)
        df.rename(columns={embedding_column_name: "embedding"}, inplace=True) 
        embeddings.append(df)
    
    # Concatenate all the embeddings into one DF.
    print("Concating embeddings")
    embeddings_df = pd.concat(embeddings)

    # Remove duplicate items for value in embedding_space_unique_food_column per dataset
    print("Creating unique embeddings")
    unique_embeddings_df = embeddings_df.groupby(level=[0, 2]).head(1)
    

    # unique_embeddings_df.sort_index(inplace=True)

    # run cosine similarity on all the embeddings
    print("Calculating cosine similarity")
    distances = pd.DataFrame(np.stack(cosine_similarity(unique_embeddings_df.embedding.tolist())), index=unique_embeddings_df.index, columns=unique_embeddings_df.index)

    distances.sort_index(axis=0, inplace=True)
    distances.sort_index(axis=1, inplace=True)

    print(f"Distance matrix dimensions: {distances.shape}")
    
    # Get the top n matches for each item in each dataset
    print(f"Retrieving the top {top_n_matches} matches for each item in each dataset")
    
    # Loop through each dataset to generate the top_n_matches.parquet file
    for dataset1 in datasets:
        for dataset2 in datasets:
            print(f"Processing top {top_n_matches} matches between {dataset1} and {dataset2}...")

            # Define paths and create directories if necessary
            top_n_match_path = Paths(dataset1, dataset2).get_top_matches_path(embedding_space, top_n_matches)

            # Skip if calculation of the comparison has already been done unless overwrite is True
            if not overwrite and check_s3_path_exists("", top_n_match_path):
                print(f"Top {top_n_matches} matches already exist for {dataset1} and {dataset2}. Skipping...")
                continue

            results = []
            with ThreadPoolExecutor() as executor:
                future_to_item = {
                    executor.submit(get_top_n_matches_for_item, item_index, distances, top_n_matches, dataset2): item_index
                    for item_index in distances.index if item_index[0] == dataset1
                }
                
                for future in tqdm(as_completed(future_to_item), total=len(future_to_item), desc=f"Processing {dataset1} vs {dataset2}"):
                    item_index = future_to_item[future]
                    matches = future.result()
                    for match_index, similarity in matches:
                        results.append({
                            f"item_in_{embedding_space_unique_food_column}": item_index[1],
                            column_names[dataset1]["unique_food_column"]: item_index[2],
                            f"match_in_{embedding_space_unique_food_column}": match_index[1],
                            f"match_{column_names[dataset2]['unique_food_column']}": match_index[2],
                            "similarity_score": similarity
                        })

            # Convert the results into a DataFrame
            top_n_matches_df = pd.DataFrame(results)

            # Add ranking for each match according to the similarity score
            # top_n_matches_df['rank'] = top_n_matches_df.groupby(f"item_in_{embedding_space_unique_food_column}")['similarity_score'].rank(method='first', ascending=False).astype(int)
            top_n_matches_df['rank'] = top_n_matches_df.groupby(f"item_in_{embedding_space_unique_food_column}")['similarity_score'].rank(method='first', ascending=False).astype(int)

            # Save the DataFrame as a Parquet file
            top_n_matches_df.to_parquet(top_n_match_path)

            print(f"Saved top matches to {top_n_match_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Cross Dataset Mapping")
    parser.add_argument("--datasets", nargs='+', type=str, help="List of datasets to create mapping.")
    parser.add_argument("--embedding_space", type=str, help="The embedding space to use for the comparison.")
    parser.add_argument("--top_n_matches", type=int, default=3, help="The number of top matches to retrieve for each food item from each dataset.")
    parser.add_argument("--overwrite", action="store_true", help="If set, overwrite existing files.")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    cross_dataset_mapping(args.datasets, args.embedding_space, args.top_n_matches, overwrite=args.overwrite)

# python fdc_gpt_alignment/scripts/cross_dataset_mapping.py --datasets MEXT2015 HPP SR_Legacy Zameret FNDDS --embedding_space SR_Legacy --top_n_matches 3
