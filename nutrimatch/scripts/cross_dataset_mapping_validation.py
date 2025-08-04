import argparse
import asyncio
import pandas as pd
import sys
sys.path.append('fdc_gpt_alignment/')
from Code.Utils import Paths
from Code.Utils.gpt import VALIDATION_TYPE, validate_food_items
from Code.Utils.production import check_s3_path_exists
from Code import Paths, SR_LegacyFoodItem, HPPFoodItem, HPPFoodItem, NearEastFoodItem, MEXT2015FoodItem, ZameretFoodItem, FooDBFoodItem, FNDDSFoodItem

"""
Cross-Dataset Mapping Validation

This script validates the cross-dataset mapping results by asking ChatGPT to classify each match as correct or incorrect.

Process:
1. Load the top matches for a given dataset combination.
2. Ask ChatGPT to classify each match as correct or incorrect.
3. Save the validated matches to a parquet file.

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

def cross_dataset_mapping_validation(dataset1: str, dataset2: str, embeddings_space: str, top_n: int, validation_type: VALIDATION_TYPE, overwrite: bool = False) -> None:
    """
    Validate the cross-dataset mapping results by asking ChatGPT to classify each match as correct or incorrect.
    
    Args:
        matches_path (str): Path to the cross-dataset mapping results.
    """
    print(f"Validating matches between {dataset1} and {dataset2} in the {embeddings_space} space.")
    print(f"Validation type: {validation_type.value}")
    matches_path = Paths(dataset1, dataset2).get_top_matches_path(embeddings_space, top_n)
    try:
        matches = pd.read_parquet(matches_path)
    except:
        print(f"Matches file not found at {matches_path}")
        print("-    First run cross_dataset_mapping.py to generate the matches file.")
        return

    validated_matches_path = Paths(dataset1, dataset2).get_validated_matches_path(embeddings_space, top_n, validation_type)

    if check_s3_path_exists("", validated_matches_path) and not overwrite:
        print(f"Validated matches already exist at {validated_matches_path}. Use --overwrite to overwrite.")
        return

    # Loop through each match and ask ChatGPT to classify it
    dataset1_unique_food_column = globals()[f"{dataset1}FoodItem"].unique_food_column
    dataset2_unique_food_column = globals()[f"{dataset2}FoodItem"].unique_food_column
    to_validate = list(matches[[dataset1_unique_food_column, f"match_{dataset2_unique_food_column}"]].itertuples(index=False, name=None))
    validations = asyncio.run(validate_food_items(to_validate, validation_type))
    indices, predictions = zip(*validations)
    matches['same_item'] = pd.Series(dtype=bool)
    matches.loc[list(indices), 'same_item'] = predictions

    matches.to_parquet(validated_matches_path)
    print(f"Validated matches saved at {validated_matches_path}")


def run_for_mappings(only: list[str], datasets: list[str], embedding_space: str, top_n: int, validation_type: VALIDATION_TYPE, overwrite: bool = False) -> None:
    """
    Run the validation step for multiple mappings.
    """
    if not only:
        only = datasets
    for dataset1 in only:
        for dataset2 in datasets:
            cross_dataset_mapping_validation(dataset1, dataset2, embedding_space, top_n, validation_type, overwrite)

def parse_args():
    parser = argparse.ArgumentParser(description="Cross-dataset mapping validation")
    parser.add_argument("--only", nargs='+', type=str, help="List of datasets to validate it's matches.")
    parser.add_argument("--datasets", nargs='+', type=str, help="List of datasets to create mapping.")
    parser.add_argument("--embedding_space", type=str, help="The embedding space to use for the comparison.")
    parser.add_argument("--top_n_matches", type=int, default=3, help="The number of top matches to retrieve for each food item from each dataset.")
    parser.add_argument("--validation_type", type=lambda value: VALIDATION_TYPE(value), choices=list(VALIDATION_TYPE), default='basic', help="The type of validation to perform.")
    parser.add_argument("--overwrite", action="store_true", help="If set, overwrite existing files.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_for_mappings(args.only, args.datasets, args.embedding_space, args.top_n_matches, args.validation_type, args.overwrite)


    
# python fdc_gpt_alignment/scripts/cross_dataset_mapping_validation.py --datasets HPP SR_Legacy Zameret FNDDS --embedding_space SR_Legacy --top_n_matches 3 
# python fdc_gpt_alignment/scripts/cross_dataset_mapping_validation.py --datasets MEXT2015 SR_Legacy Zameret FNDDS --embedding_space SR_Legacy --top_n_matches 3 --validation_type portions