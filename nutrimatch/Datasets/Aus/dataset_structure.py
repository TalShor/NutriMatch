from pydantic import BaseModel, Field, ValidationError, constr, validator
from typing import Optional, Type, List, Literal, ClassVar
from enum import Enum

from ...Utils.base_food_item import FoodItem
import sys
sys.path.append('../..')
from Code.Utils.gpt import get_embedding_parallel_generalised
from Code.Utils import Paths
import os



class AusFoodItem(FoodItem):
    food_name: str = Field(...,
        description="The name of the food item from the Australian food database")
    unique_food_column: ClassVar[str] = 'food_name'

    @validator('food_name')
    def check_length(cls, v):
        return FoodItem.check_lengths_base(v)
    
    def __str__(self):
        cls_str = f"Description: {self.food_name}."
        return cls_str
    
    # def simplify(self):
    #  return self.food_name
    
#creating embeddings
from pandarallel import pandarallel

def create_embeddings():
    """
    This function creates internal HPP embeddings.
    """
    reference_database = 'Aus'
    output_path= Paths(reference_database).food_items_path
    df = pd.read_parquet(output_path)
    #add columns representing the pydantic class for reference database

    reference_pydantic_class = AusFoodItem

    df[f'{reference_database}_class'] = reference_pydantic_class.df2fooditems(df)

    df[f'{reference_database}_dict'] = df[f'{reference_database}_class'].apply(lambda x: x.dict())
    df= df.reset_index()
    chunk_size = 100
    embeddings_path = Paths(reference_database, reference_database).embedding_path
    food_chunks = [df[i : i + chunk_size] for i in range(0, df.shape[0], chunk_size)]

    pandarallel.initialize(progress_bar=True, nb_workers=64)
    for chunk in  food_chunks[1:60]:
        chunk_number = int(chunk.index[0]/chunk_size)
        print(chunk_number)
        chunk[f'{reference_database}_embeddings'] = chunk[f'{reference_database}_dict'].parallel_apply(lambda x: get_embedding_parallel_generalised(x,reference_pydantic_class))
        chunk.drop(columns=[f'{reference_database}_class'], inplace=True)
        
        path = f'{embeddings_path}chunk_{chunk_number}.parquet'
        print(path)
        chunk.to_parquet(path)