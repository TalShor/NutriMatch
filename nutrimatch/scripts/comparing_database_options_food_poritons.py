import pandas as pd
import sys
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
import enum
from pandarallel import pandarallel

sys.path.append('fdc_gpt_alignment/')
from Code import get_secret, Paths


api_key =  get_secret()['open_ai_key']
client = instructor.patch(OpenAI(api_key=api_key))


# function likely does not need to be run again, keeping it here for reference

fpath_database_comparison_df = 's3://datasets-development/diet/japan_registry/evaluating_databases/external_db_matched_to_mext2015.parquet'
def merging_datasets():
    """
    Merges the results of matching MEXT2015  with the 3 databases: SR_Legacy, FNDDS,  Zameret. One time process

    Returns:
        merged_df (DataFrame): The merged dataframe containing the results from all 4 databases.
    """
    food_items = pd.read_parquet(Paths('SR_Legacy').food_items_path)
    sr_legacy_match = pd.read_parquet(Paths('SR_Legacy', 'MEXT2015').top_match_path)
    fndds_match = pd.read_parquet(Paths('FNDDS', 'MEXT2015').top_match_path)
    zameret_match = pd.read_parquet(Paths('Zameret', 'MEXT2015').top_match_path)

    sr_legacy_match = sr_legacy_match \
            .rename(columns={'description': 'sr_legacy_food_item', 'similarity_score': 'sr_legacy_similarity_score'}) \
            .reset_index().drop_duplicates( subset = 'food_item').set_index('food_item')


    zameret_match = zameret_match \
                .rename(columns={'english_name': 'zameret_food_item', 'similarity_score': 'zameret_similarity_score'})\
                .reset_index().drop_duplicates(subset = 'food_item') \
                .set_index('food_item')

    fndds_match = fndds_match \
        .rename(columns={'description': 'fndds_food_item', 'similarity_score': 'fndds_similarity_score'}) \
        .reset_index().drop_duplicates(subset  ='food_item').set_index('food_item')

    merged_df = sr_legacy_match.join(fndds_match).join(zameret_match)

    output = fpath_database_comparison_df
    merged_df.to_parquet(output)
    
    return merged_df


class Labels(str, enum.Enum):
    """Enumeration for evaluation for database option."""

    sr_legacy= "sr_legacy"
    fndds = "FNDDS"
    zameret = "Zameret"
    no_relevant_match = "no_relevant_match"

class FoodMatchEvaluation(BaseModel):
    """
    Class for a single class label prediction.
    """

    database_label: Labels = Field(..., 
        description="Specifies the database to which the food item closely aligns.")
    reason: str = Field(..., 
        description="Explains the reason for choosing the match with a specific database, or indicates why no appropriate match was identified.")


# Apply the patch to the OpenAI client
# enables response_model keyword
def classify_internal_external(food_item_internal: str, food_items_external: list) -> FoodMatchEvaluation:
    """
    Determine the best match for a food item among three external food items in English.
    Select the database whose item closely matches the internal one in relation to portion size matching, 
    overall similarity, and state (e.g., cooked or raw), with a focus on the usability of external mapping tables 
    for unit conversions (e.g., gram to cup).

    Args:
    - food_item_internal (str): The food item from the internal database.
    - food_items_external (List[str]): List of three food items from the external databases.

    Returns:
    - FoodMatchEvaluation: The best match evaluation.
    """

    prompt = (
        f"As a registered dietitian, evaluate the similarity of a food item "
        f"to three external food items from different databases. "
        f"Internal item: '{food_item_internal}'. "
        f"External items: sr_legacy - '{food_items_external[0]}', "
        f"FNDDS - '{food_items_external[1]}', Zameret - '{food_items_external[2]}'. "
        f"Focus on factors such similarity of items, usability of external unit conversion tables (e.g., gram to cup), "
        f"and whether the item is cooked or raw. Determine the best match based on these criteria. If two or more options are equally similar, "
        f"prioritize sr_legacy. Provide a brief explanation for your choice. If no relevant match is found, indicate this."
    )
    
    client = instructor.patch(OpenAI(api_key=api_key))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=FoodMatchEvaluation,
        messages=[{"role": "system", "content": prompt}],
    )

    database_choice = response.database_label.value
    reason = response.reason

    return database_choice, reason


def apply_match(row: pd.Series) -> tuple:
    """
    Applies the classify_internal_external function to a row of a DataFrame.

    Args:
    - row (pd.Series): A row from the DataFrame containing food items.

    Returns:
    - tuple: A tuple containing the database choice and the reason for that choice.
    """
    food_item_internal = row['food_item']  # Accessing by column name
    food_items_external = [
        row['sr_legacy_food_item'], 
         row['fndds_food_item'],
        row['zameret_food_item']
    ]

    database_choice, reason = classify_internal_external(food_item_internal, food_items_external)
    return database_choice, reason



def apply_match_to_df() -> pd.DataFrame:
    """
    Applies the 'apply_match' function to each row o`f the 'food_registry' DataFrame in parallel chunks.
    Saves the processed chunks to separate Parquet files. This chooses the best match for each food item in the registry.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    # Initialize pandarallel
    nb_workers = 64
    pandarallel.initialize(nb_workers=nb_workers, progress_bar=True)

    # Define the chunk size
    chunk_size = 100
    start_chunk = 20 # need to do chunk 9
    end_chunk =  24 # can change this as needed


    output_path = 's3://datasets-development/diet/japan_registry/evaluating_databases/database_choice/'
    food_registry = pd.read_parquet(fpath_database_comparison_df)
    # Loop over each chunk using the chunk number
    for chunk_num in range(start_chunk, end_chunk):
        start_idx = chunk_num * chunk_size
        end_idx = start_idx + chunk_size
        chunk = food_registry.iloc[start_idx:end_idx]

        chunk[['database_choice', 'reason']] = chunk.parallel_apply(apply_match, axis=1, result_type='expand')

        print(f"Processed chunk {chunk_num}.")
        # Save the processed chunk to a CSV file
        chunk.to_parquet(f'{output_path}chunk_{chunk_num}.parquet')


if __name__ == '__main__':
    apply_match_to_df()

# python fdc_gpt_alignment/scripts/comparing_database_options_food_poritons.py
