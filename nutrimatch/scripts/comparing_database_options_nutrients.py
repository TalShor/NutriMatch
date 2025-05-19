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
def adding_FNDDS_to_registry():
    """
    Adds FNDDS data to the food registry.

    Returns:
    - DataFrame: Updated food registry with FNDDS data.
    """
    similarity_results = pd.read_parquet('s3://datasets-development/diet/USDA_FNDDS/hpp_foodb/similarity_results/')
    fndds = pd.read_parquet('s3://datasets-development/diet/USDA_FNDDS/text/fndds_foods_with_categories_and_embeddings.parquet')
    food_registry  = pd.read_csv('s3://datasets-development/diet/food_registry/food_registry/v1/food_registry_sr_legacy_foodb_only.csv', index=False)

    similarity_results = similarity_results[['food_id', 'description', 'similarity_score', 'fndds_food_category']]
    similarity_results_joined = similarity_results.merge(fndds[['description', 'food_subcategory']], on='description')
    cols_renamed = {'food_subcategory': 'fndds__food_subcategory','description':'fndds__description', 'similarity_score': 'fndds__similarity_score', 'fndds_food_category': 'fndds__food_category'}
    similarity_results_joined.rename(columns = cols_renamed, inplace = True)

    food_registry = food_registry.set_index('food_id').join(similarity_results_joined.set_index('food_id'), how='left')
    first_cols = ['display_name', 'product_name', 'short_food_name', 'food_category',
                'sr_legacy__description', 'foodb__name','fndds__description', 
                'sr_legacy__similarity_score', 'foodb__similarity_score', 'fndds__similarity_score', 
                'fndds__food_category', 'fndds__food_subcategory']
    rest_cols = [col for col in food_registry.columns if col not in first_cols]
    food_registry = food_registry[first_cols + rest_cols]

    return food_registry


def merging_datasets():
    """
    Merges the results from the 4 databases: SR_Legacy, FNDDS, FooDB, and Zameret. One time process.

    Returns:
        merged_df (DataFrame): The merged dataframe containing the results from all 4 databases.
    """
    top_match_SR_legacy = pd.read_parquet(Paths('SR_Legacy', 'HPP').top_match_path)
    top_match_SR_legacy = top_match_SR_legacy.set_index('hpp_display_name')[['description', 'similarity', 'hpp_product_name', 'hpp_short_food_name', 'hpp_food_category']]
    top_match_SR_legacy =  top_match_SR_legacy.rename(columns = {'similarity': 'SR_Legacy__similartiy_score', 'description':'SR_Legacy__description'})

    top_match_FNDDS = pd.read_parquet(Paths('FNDDS', 'HPP').top_match_path).set_index('display_name')
    top_match_FNDDS.index.name = 'hpp_display_name'
    top_match_FNDDS = top_match_FNDDS[['description', 'similarity_score']].rename(columns = {'description': 'FNDDS__description', 'similarity_score': 'FNDDS__similarity_score'})

    top_match_FooDB = pd.read_parquet(Paths('FooDB', 'HPP').top_match_path)
    top_match_FooDB = top_match_FooDB.rename(columns = {'display_name': 'hpp_display_name', 'name': 'FooDB__name', 'similarity_score': 'FooDB__similarity_score'})
    top_match_FooDB = top_match_FooDB.set_index('hpp_display_name')[['FooDB__name', 'FooDB__similarity_score']]


    top_match_Zameret = pd.read_parquet(Paths('Zameret', 'HPP').top_match_path)
    top_match_Zameret  = top_match_Zameret.reset_index().set_index('hpp__hebrew_name')
    top_match_Zameret.index.name = 'hpp_display_name'
    top_match_Zameret = top_match_Zameret.rename(columns = {'english_name': 'Zameret__english_name', 'similarity_score': 'Zameret__similarity_score', 'hebrew_name':'Zameret__hebrew_name'})

    merged_df = top_match_SR_legacy.join(top_match_FNDDS).join(top_match_FooDB).join(top_match_Zameret)

    merged_df.to_csv(Paths('HPP').hpp_registry_with_top_matches)
    
    return merged_df


class Labels(str, enum.Enum):
    """Enumeration for evaluation for database option."""

    sr_legacy= "sr_legacy"
    foodb = "FooDB"
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

def classify_internal_external(food_item_internal:str, food_items_external:list) -> FoodMatchEvaluation:
    """
    Determine the best match for an internal food item in Hebrew against three external food items in English.
    Select the database whose item closely matches the internal one in nutrition and overall similarity.

    Args:
    - food_item_internal (str): The food item from the internal database.
    - food_items_external (List[str]): List of three food items from the external databases.

    Returns:
    - FoodMatchEvaluation: The best match evaluation.
    """

    prompt = (
    f"As a registered dietician, evaluate the similarity of an internal Hebrew food item "
    f"against four food items from different databases. "
    f"Internal item: '{food_item_internal}'. "
    f"External items: sr_legacy - '{food_items_external[0]}', "
    f"FooDB - '{food_items_external[1]}', FNDDS - '{food_items_external[2]}', Zameret - '{food_items_external[3]}'."
    f"Determine the best match based on nutrition and similarity. If two or more options seem equally similar, "
    f"prioritize sr_legacy. Briefly explain your choice in a short sentence. If no relevant match is found, indicate so."
)
    client = instructor.patch(OpenAI(api_key=api_key))

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
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
      It should have 'display_name', 'sr_legacy__description', 'foodb__name','fndds__description',  columns.

    Returns:
    - tuple: A tuple containing the database choice and the reason for that choice.
    """
    food_item_internal = row['display_name']  # Accessing by column name
    food_items_external = [
        row['SR_Legacy__description'],  # Accessing by column name
        row['FooDB__name'],             # Accessing by column name
        row['FNDDS__description'] ,
        row['Zameret__english_name']     # Accessing by column name
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
    nb_workers = 32
    pandarallel.initialize(nb_workers=nb_workers, progress_bar=True)

    # Define the chunk size
    chunk_size = 100
    start_chunk = 9 # need to do chunk 9
    end_chunk =  10 # can change this as needed

    output_path = 's3://datasets-development/diet/food_registry/HPP/evaluating_databases/'
    food_registry = pd.read_csv(Paths('HPP').hpp_registry_with_top_matches)
    # Loop over each chunk using the chunk number
    for chunk_num in range(start_chunk, end_chunk):
        start_idx = chunk_num * chunk_size
        end_idx = start_idx + chunk_size
        chunk = food_registry.iloc[start_idx:end_idx]

        chunk[['database_choice', 'reason']] = chunk.parallel_apply(apply_match, axis=1, result_type='expand')

        print(f"Processed chunk {chunk_num}.")
        # Save the processed chunk to a CSV file
        chunk.to_parquet(f'{output_path}food_registry_chunk_{chunk_num}.parquet')


if __name__ == '__main__':
    apply_match_to_df()

# python fdc_gpt_alignment/scripts/comparing_database_options.py
