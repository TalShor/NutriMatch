import os

from Code.Utils.gpt import VALIDATION_TYPE
class Paths:
    def __init__(self, dataset_name = None, other_dataset_name = None, temp_folder = None) -> None:
        # Base path for dataset storage
        self.base_path = 's3://datasets-development/diet/dataset_alignment/'
        self.dataset_path = os.path.join(self.base_path, dataset_name)
        self.raw_data_path = os.path.join(self.dataset_path, "RawData")
        self.usable_data_path = os.path.join(self.dataset_path, "UsableData")
        self.food_items_path = os.path.join(self.usable_data_path, "FoodItems.parquet")
        self.food_portions_path = os.path.join(self.usable_data_path, "FoodPortions.parquet")
        self.nutrients_path = os.path.join(self.usable_data_path, "Nutrients.parquet")
        #TODO - check if we need more granularity when defining this path below - i,e if we compare diff datasets
        self.evaluating_databases_path = 's3://datasets-development/diet/food_registry/evaluating_databases/'
        self.food_registry = 's3://datasets-development/diet/food_registry/food_registry/v1/food_registry_with_similarities.csv'
        self.hpp_registry_with_top_matches = 's3://datasets-development/diet/food_registry/food_registry/v2/full_food_registry_with_similarities.csv'


        # Check if a temporary folder path is provided for specific paths
        self.base_path_for_temp = temp_folder if temp_folder else self.dataset_path

        self.embedding_path = os.path.join(self.base_path_for_temp, f"Embeddings/{other_dataset_name}/")
        self.distance_matrix_path = os.path.join(self.base_path_for_temp, f"DistanceMatrix/{other_dataset_name}/")
        self.top_match_path = os.path.join(self.base_path_for_temp, f"TopMatch/{other_dataset_name}/")
        self.database_comparison_path = os.path.join(self.base_path_for_temp, f"DatabaseComparison/{other_dataset_name}/")
    
    def get_top_matches_path(self, embedding_space: str, top_n: int = 3) -> str:
        return os.path.join(self.database_comparison_path, f"top_n_matches/{embedding_space}/top_{top_n}_matches.parquet")
    
    def get_validated_matches_path(self, embedding_space: str, top_n: int = 3, validation_type: VALIDATION_TYPE = VALIDATION_TYPE.VALIDATED_MATCHES) -> str:
        return os.path.join(self.database_comparison_path, f"{validation_type.get_directory_name()}/{embedding_space}/top_{top_n}_matches.parquet")


# #TODO read function and return function


# # Example usage
# foo_db_paths = Paths("FooDB")
# fndds_paths = Paths("FNDDS")
# sr_legacy_paths = Paths("SR_Legacy")

# print("FooDB Raw Data Path:", foo_db_paths.raw_data_path)
# print("FNDDS Usable Data Path:", fndds_paths.usable_data_path)
# print("SR_Legacy Translation Path for DB1:", sr_legacy_paths.get_translation_path("OtherDB1"))
