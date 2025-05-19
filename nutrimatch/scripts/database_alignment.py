import pandas as pd
import sys
import argparse

sys.path.append('fdc_gpt_alignment/')

from Code import  running_all_chunks, Paths, FNDDSFoodItem, HPPFoodItem, FooDBFoodItem, SR_LegacyFoodItem, ZameretFoodItem, NearEastFoodItem, MEXT2015FoodItem, UAEChatGPTFoodItem, BahrainFoodItem, AusFoodItem

#TODO globals  - function - get 
def matching_process(args):
    """
    Perform matching process for external dataset.

    Parameters:
    args (argparse.Namespace): Arguments.

    Returns:
    None
    """

    reference_database = args.reference_db
    external_database = args.external_db

    reference_pydantic_class = globals()[reference_database + 'FoodItem']
    external_pydantic_class = globals()[external_database + 'FoodItem']

    food_items_reference_database_path = Paths(reference_database).food_items_path
    food_items_reference_database = pd.read_parquet(food_items_reference_database_path )
    
    external_database_embeddings_path = Paths(external_database, external_database).embedding_path 
    external_database_df = pd.read_parquet(external_database_embeddings_path)
    
    #add columns representing the pydantic class for reference database
    food_items_reference_database[f'{reference_database}_class'] = reference_pydantic_class.df2fooditems(food_items_reference_database)
 
    # when temp folder is None it reverts to actual paths
    temp_folder = args.temp_folder
    top_match_path  =Paths(external_database, reference_database,temp_folder  ).top_match_path
    embedding_path   = Paths(external_database, reference_database, temp_folder ).embedding_path
    distance_matrix_path = Paths(external_database, reference_database, temp_folder ).distance_matrix_path

    running_all_chunks(food_items_reference_database = food_items_reference_database , 
                    chunk_size = args.chunk_size, 
                    top_match_path = top_match_path ,
                    distance_matrix_path = distance_matrix_path, 
                    external_database_df =external_database_df, 
                    embedding_path = embedding_path ,external_database = 
                    external_database, reference_database =reference_database,
                    external_pydantic_class = external_pydantic_class, 
                    chunk_start = args.chunk_start, 
                    chunk_end = args.chunk_end, 
                    overwrite = args.overwrite)

    return None



def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Matching Process")
    parser.add_argument("--external_db", type = str, help = "External Database")
    parser.add_argument("--reference_db", type = str, default = 'HPP', help = "Reference Database")
    parser.add_argument("--chunk_start", type=int, default=0, help="Start index of chunk")
    parser.add_argument("--chunk_end", type=int, default=2, help="End index of chunk")
    parser.add_argument("--chunk_size", type=int, default=100, help="Size of each chunk")
    parser.add_argument("--temp_folder", type=str, default=False, help="temp base path for dataset")
    parser.add_argument("--overwrite", type=bool, default=False, help="If set to True, overwrite existing files")
    # Add other parameters here as needed

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    matching_process(args)

# python fdc_gpt_alignment/scripts/database_alignment.py  --external_db SR_Legacy   --reference_db Aus --chunk_start 6 --chunk_end 20 --chunk_size 100 

# python fdc_gpt_alignment/scripts/database_alignment.py  --external_db SR_Legacy   --reference_db FNDDS --chunk_start 0 --chunk_end 58 --chunk_size 100 

# python fdc_gpt_alignment/scripts/database_alignment.py  --external_db Zameret  --reference_db HPP --chunk_start 0 --chunk_end 1 --chunk_size 100 

# python fdc_gpt_alignment/scripts/database_alignment.py  --external_db HPP --reference_db Zameret  --chunk_start 0 --chunk_end 1 --chunk_size 100 

# python fdc_gpt_alignment/scripts/database_alignment.py  --external_db SR_Legacy --reference_db Bahrain  --chunk_start 0 --chunk_end 301 --chunk_size 2
# python fdc_gpt_alignment/scripts/database_alignment.py  --external_db SR_Legacy  --reference_db UAEChatGPT --chunk_start 0 --chunk_end 808 --chunk_size 1 --overwrite True
# python fdc_gpt_alignment/scripts/database_alignment.py  --external_db SR_Legacy --reference_db NearEast  --chunk_start 0 --chunk_end 15 --chunk_size 100 --overwrite True
# python fdc_gpt_alignment/scripts/database_alignment.py  --external_db HPP --reference_db Zameret  --chunk_start 0 --chunk_end 30 --chunk_size 100 
# python fdc_gpt_alignment/scripts/database_alignment.py  --external_db Zameret  --reference_db MEXT2015 --chunk_start 0 --chunk_end 1 --chunk_size 100 
