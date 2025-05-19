import pandas as pd
import sys
sys.path.append('../..')
from Code import Paths, ZameretFoodItem, SR_LegacyFoodItem, HPPFoodItem
from pandarallel import pandarallel

from Code.Utils.gpt import get_embedding_parallel_generalised
from Code.Utils.base_food_item import FoodItem
from Code.Utils.production import convert_class_to_external


#creating embeddings
def create_embeddings():
    """
    This function creates internal HPP embeddings.
    """
    reference_database = 'HPP'
    output_path= Paths(reference_database).food_items_path
    output_path= Paths(reference_database).food_items_path
    df = pd.read_parquet(output_path)
    #add columns representing the pydantic class for reference database

    reference_pydantic_class = HPPFoodItem

    df[f'{reference_database}_class'] = reference_pydantic_class.df2fooditems(df)

    df[f'{reference_database}_dict'] = df[f'{reference_database}_class'].apply(lambda x: x.dict())
    df= df.reset_index()
    chunk_size = 100
    embeddings_path = Paths('HPP', 'HPP').embedding_path
    food_chunks = [df[i : i + chunk_size] for i in range(0, df.shape[0], chunk_size)]

    pandarallel.initialize(progress_bar=True, nb_workers=64)
    for chunk in  food_chunks[1:77]:
        chunk_number = int(chunk.index[0]/chunk_size)
        print(chunk_number)
        chunk['HPP_embeddings'] = chunk[f'{reference_database}_dict'].parallel_apply(lambda x: get_embedding_parallel_generalised(x,HPPFoodItem))
        chunk.drop(columns=[f'{reference_database}_class'], inplace=True)
        chunk.to_parquet(f'{embeddings_path}chunk_{chunk_number}.parquet')