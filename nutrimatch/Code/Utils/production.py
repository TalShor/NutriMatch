import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from botocore.exceptions import  ClientError
import numpy as np
from enum import Enum
from pandarallel import pandarallel
import boto3

from .base_food_item import FoodItem
from .gpt import  get_translation_parallel ,get_embedding_parallel_generalised


# Assuming 'df' is your DataFrame
def rename_duplicates(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique(): 
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df


def running_all_chunks(
                    food_items_reference_database: pd.DataFrame, 
                    chunk_size: int,
                    top_match_path: str, 
                    distance_matrix_path:str, 
                    external_database_df: pd.DataFrame, 
                    embedding_path: str, 
                    external_database: str,
                    reference_database: str,
                    external_pydantic_class:FoodItem,
                    chunk_start:int, 
                    chunk_end:int, 
                    overwrite:bool = False) -> None:
    """
    Run matching process for all chunks.

    Parameters:
    - food_items_reference_database (pd.DataFrame): Reference database with food items.
    - chunk_size (int): Size of each chunk.
    - top_match_path (str): Path to save similarity results.
    - distance_matrix_path (str): Path to save distance matrix results.
    - external_database_df (pd.DataFrame): DataFrame of the external database.
    - embedding_path (str): Path to save embedding results.
    - external_database (str): Name of the external database.
    - reference_database (str): Name of the reference database.
    - external_pydantic_class (FoodItem): Pydantic class for the external database.
    - chunk_start (int): Starting index of the chunk.
    - chunk_end (int): Ending index of the chunk.
    - overwrite (bool): Flag to overwrite existing results. Default is False.

    Returns:
    None
    """
    food_chunks = [food_items_reference_database[i : i + chunk_size] for i in range(0, food_items_reference_database.shape[0], chunk_size)]

    for chunk in food_chunks[chunk_start: chunk_end]:
        matching_process_per_chunk(chunk=chunk, top_match_path=top_match_path, external_database_df=external_database_df, base_embedding_path=embedding_path,distance_matrix_path = distance_matrix_path,  workers=32, external_database=external_database,reference_database=reference_database,  chunk_size=chunk_size, external_pydantic_class=external_pydantic_class,  overwrite =  overwrite)

    print('done running all chunks')
    return None


def matching_process_per_chunk(
                chunk: pd.DataFrame, 
                top_match_path: str,
                base_embedding_path: str,
                distance_matrix_path:str,  
                external_database:str , 
                reference_database:str, 
                external_database_df:pd.DataFrame, 
                external_pydantic_class:FoodItem, 
                workers: int = 32, 
                chunk_size: int = 100, 
                bucket_name: str = "datasets-development", 
                overwrite:bool = False) -> None:
    """
    Perform matching process per chunk for Foodb dataset.

    Parameters:
    chunk (pandas.DataFrame): Chunk data.
    top_match_path (str): Path to save similarity results.
    hpp_foodb_class_path (str): Path to save HPP to Foodb class conversion results.
    workers (int): Number of workers for parallel processing. Default is 32.
    database (str): Database name. Default is 'foodb'.
    chunk_size (int): Size of each chunk. Default is 100.
    bucket_name (str): Name of the S3 bucket. Default is "datasets-development".

    Returns:
    None
    """
  
    pandarallel.initialize(progress_bar=True, nb_workers=workers)
    
    chunk_number = chunk.index[0] // chunk_size
    results_path = f'{top_match_path}chunk_{chunk_number}.parquet'
    embedding_path = f'{base_embedding_path}chunk_{chunk_number}.parquet'
    distance_matrix_path = f'{distance_matrix_path}chunk_{chunk_number}.parquet'

    exists = check_s3_path_exists(bucket_name, results_path)
    
    if exists and not overwrite: #if file exists and overwrite is False then do not run
        print(f'chunk_{chunk_number} already exists')
    else:

        # converting reference class to external class 
        chunk_final, chunk_phase_1, chunk_redone = convert_class_to_external(chunk=chunk,
                                                                            external_database=external_database, 
                                                                            reference_database=reference_database,
                                                                            external_class=external_pydantic_class)
        
        food_item_col_reference = chunk_final[f'{reference_database}_class'].values[0].unique_food_column

        external_database_df[f'{external_database}_class'] = external_pydantic_class.df2fooditems(external_database_df.reset_index())
        food_item_col_external = external_database_df[f'{external_database}_class'].values[0].unique_food_column

        # getting embeddings
        embedding_results = creating_embedding_results(
            chunk_final=chunk_final,
            chunk_phase_1=chunk_phase_1,
            chunk_redone=chunk_redone,
            external_database=external_database,
            reference_database=reference_database,
            YourFoodItemClass=external_pydantic_class,
            food_item_col_reference=food_item_col_reference
        )

        # calculate the similarities
        ## using config to retriev the unique column for the external and reference database
        
        distance_matrix_df, top_matches_with_score_df = creating_similarties_results(
            chunk_final=chunk_final,
            external_database_df=external_database_df,
            external_database=external_database,
            reference_database=reference_database,
            food_item_col_external=food_item_col_external,
            food_item_col_reference=food_item_col_reference
        )

        print(f'chunk_{chunk_number} done')
        # saving results
        print(f'saving results to {embedding_path}, {results_path} and {distance_matrix_path}')
        embedding_results.to_parquet(embedding_path)
        top_matches_with_score_df.to_parquet(results_path)

        # Call the function to rename duplicates
        distance_matrix_df = rename_duplicates( distance_matrix_df)
        distance_matrix_df.to_parquet(distance_matrix_path)
    


def convert_class_to_external(chunk: pd.DataFrame, 
                              reference_database: str, 
                              external_database: str, 
                              external_class: FoodItem):
    """
    Convert items from one class format to another and simplify descriptions.

    Parameters:
    - chunk (pd.DataFrame): The data frame containing the items.
    - reference_database (str): The name of the database from which items are being converted.
    - external_database (str): The name of the database to which items are being converted.
    - external_class (FoodItem): The class type to which items will be converted.
    - simplify_description (Callable[[FoodItem], FoodItem]): A function that takes a FoodItem and returns a simplified version of it.


    Returns:
    pd.DataFrame: The updated DataFrame with converted class information and necessary modifications.
    """

    # Translate the class from reference database to external database
    dictionary_class_column = f'{reference_database}_to_{external_database}_dict'

    chunk[dictionary_class_column] = chunk[f'{reference_database}_class'].parallel_apply(
        lambda x: get_translation_parallel(x, external_class))
    
    chunk_phase_1 = chunk.copy()
    
    # Handle discrepancies in items
    chunk['item_discrepancy'] = chunk.apply(
        lambda x: x[dictionary_class_column]['food_item_discrepancy'] if x[dictionary_class_column] is not None else None, axis=1)

    # Print number of items with discrepancy
    discrepancy_mask = chunk['item_discrepancy'].isin(['not same item', 'likely not same item'])
    print(f'Number of items with discrepancy: {discrepancy_mask.sum()}')

    # Modify the description for items with discrepancies
    
    chunk.loc[discrepancy_mask, f'{reference_database}_class'] = chunk.loc[discrepancy_mask, f'{reference_database}_class'].apply(lambda x: x.simplify())

    # Re-translate the modified items
    chunk_redone = chunk.loc[discrepancy_mask].copy()
    chunk_redone[dictionary_class_column] = chunk_redone[f'{reference_database}_class'].apply(
        lambda x: get_translation_parallel(x, external_class))
    chunk.loc[discrepancy_mask, dictionary_class_column] = chunk_redone[dictionary_class_column]

    return chunk.copy(),  chunk_phase_1.copy() , chunk_redone.copy()


def create_chunk_both_versions(chunk_redone: pd.DataFrame, chunk_final: pd.DataFrame, chunk_phase_1: pd.DataFrame, external_database: str, reference_database:str,  food_item_col_reference:str) -> pd.DataFrame:
    """
    Create a chunk that combines both versions of data.

    Parameters:
    - chunk_redone (pd.DataFrame): The chunk with modified items.
    - chunk_final (pd.DataFrame): The final chunk of data.
    - chunk_phase_1 (pd.DataFrame): The chunk from the first phase with discrepancies.
    - external_database (str): The name of the external database.
    - reference_database (str): The name of the reference database.
    - food_item_col_reference (str): The name of the column containing food item references.

    Returns:
    - chunk_df (pd.DataFrame): The combined chunk with both versions of data.
    """

    food_item_column = food_item_col_reference
    print('col is ' + food_item_column)
    print(chunk_redone.columns)
    food_id_columns_redone = chunk_redone[food_item_column].to_list()

    chunk_with_discrepancy = chunk_phase_1[chunk_phase_1[food_item_column].isin(food_id_columns_redone)]

    print(f'foods with discrepancy: {chunk_with_discrepancy[food_item_column].to_list()}')
    chunk_final['phase'] = 'final'
    chunk_with_discrepancy['phase'] = 'first_phase_discrepancy'

    both_versions_chunk = pd.concat([chunk_final, chunk_with_discrepancy])
    
    database_class_dict_column = f'{reference_database}_to_{external_database}_dict'

    # Converting dictionary (class of food database) to DataFrame
    class_as_df = both_versions_chunk[database_class_dict_column].apply(pd.Series)
    class_as_df = class_as_df.add_prefix(f'{external_database}__')
    # Adding the columns from class onto original
    chunk_df = both_versions_chunk.join(class_as_df)

    return chunk_df


def creating_embedding_results(
        chunk_final: pd.DataFrame, 
        chunk_phase_1: pd.DataFrame, 
        chunk_redone: pd.DataFrame,
        external_database:str , 
        reference_database:str,
        YourFoodItemClass:FoodItem, 
        food_item_col_reference:str ):
    """
    Create embedding results for Foodb dataset.

    Parameters:
    chunk_final (pandas.DataFrame): Final chunk data.
    chunk_phase_1 (pandas.DataFrame): Phase 1 chunk data.
    chunk_redone (pandas.DataFrame): Redone chunk data.
    database (str): Database name.

    Returns:
    pandas.DataFrame: Embedding results.
    """
    reference_external_string = f'{reference_database}_to_{external_database}'
    chunk_final[f'{reference_external_string}_embedding'] = chunk_final[f'{reference_external_string}_dict'].parallel_apply(
    lambda food_item: get_embedding_parallel_generalised(food_item, YourFoodItemClass)
)

    embedding_results = create_chunk_both_versions(chunk_redone, chunk_final, chunk_phase_1, external_database, reference_database,  food_item_col_reference =  food_item_col_reference)
    embedding_results = embedding_results.drop(columns=[f'{reference_external_string}_dict', f'{reference_database}_class', 'item_discrepancy'])

    # print(embedding_results.columns)
    for column in embedding_results.columns:
        non_na_values = embedding_results[column].dropna()
        if not non_na_values.empty:
            if isinstance(non_na_values.iloc[0], Enum):
                print(f'{column} is of type Enum')
                embedding_results[column] = embedding_results[column].apply(lambda x: x.value if isinstance(x, Enum) else x)
        else:
            print(f'{column} is empty or all NaNs')

    return embedding_results


def creating_similarties_results(
        chunk_final: pd.DataFrame, 
        external_database_df: pd.DataFrame, 
        external_database: str,
        reference_database:str, 
        food_item_col_external:str, 
        food_item_col_reference:str ) -> pd.DataFrame:
    """
    Create similarity results for Foodb dataset.

    Parameters:
    chunk_final (pandas.DataFrame): Final chunk data.
    foodb (pandas.DataFrame): Foodb data.
    database (str): Database name.

    Returns:
    pandas.DataFrame: Similarity results.
    """
    chunk_embeddings = np.stack(chunk_final[f'{reference_database}_to_{external_database}_embedding'])
    external_database_embeddings = np.stack(external_database_df[f'{external_database}_embedding'])
    # TODO = database col need to generalise

    similarity_matrix = cosine_similarity(chunk_embeddings, external_database_embeddings )

    # Create DataFrame from the similarity matrix
    distance_matrix_df = pd.DataFrame(similarity_matrix, index=chunk_final[food_item_col_reference], columns=external_database_df [food_item_col_external])

    # Find the top match and its score for each display_name
    top_matches_with_score_df = distance_matrix_df.apply(lambda row: pd.Series([row.idxmax(), row.max()], index=[food_item_col_external, 'similarity_score']), axis=1)



    return  distance_matrix_df , top_matches_with_score_df


def check_s3_path_exists (bucket_name: str, file_path: str):
    """
    Checks if a file exists in an S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.
        file_path (str): The path of the file in the S3 bucket.

    Returns:
        bool: True if the file exists, False otherwise.

    Raises:
        Exception: If credentials are not found for the given profile.
    """
    # Check if file_path is a full S3 URI and split to get bucket_name and file_path
    if file_path.startswith("s3://"):
        parts = file_path.split("/", 3)
        bucket_name = parts[2]
        file_path = parts[3] if len(parts) > 3 else ""

    # session = boto3.session.Session(profile_name=profile_name)
    s3 = boto3.client('s3')

    try:
        s3.head_object(Bucket=bucket_name, Key=file_path)
        return True
    except ClientError as e:
        return False



