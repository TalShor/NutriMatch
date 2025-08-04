import pandas as pd
import os
from typing import List
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Usage
file_path = 's3://datasets-development/diet/dataset_alignment/MEXT/RawData/STANDARD TABLES OF FOOD COMPOSITION IN JAPAN - 2015 - (Seventh Revised Edition).xlsx'
def process_MEXT_raw_data(file_path = file_path):
    # Load the Excel file
    raw_table = pd.read_excel(file_path)
    raw_table.columns = raw_table.iloc[4]
    raw_table = raw_table[5:]
    raw_table.columns = raw_table.columns.str.replace('\n', '', regex=True)

    # Extract meta data
    meta_data = raw_table.head(2).T
    meta_data.columns = meta_data.iloc[3]
    meta_data = meta_data[4:]
    meta_data = meta_data.reset_index()
    meta_data.columns = ['descriptions', 'tag_names', 'unit']

    # Continue processing raw_table
    raw_table = raw_table[2:]
    raw_table = raw_table.rename(columns={'Food and Description': 'food_item'})

    # Define food categories
    food_categories = [
        "Grains", "Potatoes and Starches", "Sugars and Sweeteners", "Pulses", "Nuts and Seeds",
        "Vegetables", "Fruit", "Mushrooms", "Algae", "Fish and Shellfish", "Meat", "Eggs",
        "Dairy Products", "Fats and Oils", "Confectioneries", "Beverages", "Seasonings and Spices",
        "Prepared and Processed Foods"
    ]

    codes = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18"]

    # Create DataFrame for food categories
    food_category_df = pd.DataFrame({'food_category': food_categories, 'Food Group': codes})
    food_category_df['Food Group'] = food_category_df['Food Group'].astype(str)

    # Merge with raw table
    merged_table = raw_table.set_index('Food Group').join(food_category_df.set_index('Food Group'), on='Food Group')
    merged_table = merged_table[['Item No.', 'food_category'] + list(merged_table.columns[1:-1])]

    # Create a DataFrame for food items
    food_items_df = merged_table[['Item No.', 'food_category', 'food_item']].copy()

    #ad hoc change - english name was --
    # food_items_df.loc[985].food_item = 'Makombu, sun-dried'

    return food_items_df, meta_data



# def compute_similarity_and_save(embeddings_path_1, embeddings_path_2, output_path, reference_col='food_item', external_col='description'):
#     # Load embeddings
#     embeddings_df_1 = pd.read_parquet(embeddings_path_1)
#     embeddings_df_2 = pd.read_parquet(embeddings_path_2)
    
#     # Convert embeddings to numpy arrays for cosine similarity
#     embeddings_1 = np.stack(embeddings_df_1['MEXT_to_SR_Legacy_embedding'])
#     embeddings_2 = np.stack(embeddings_df_2['FNDDS_to_SR_Legacy_embedding'])
    
#     # Compute cosine similarity matrix
#     similarity_matrix = cosine_similarity(embeddings_1, embeddings_2)
    
#     # Convert similarity matrix to DataFrame
#     distance_matrix_df = pd.DataFrame(similarity_matrix, index=embeddings_df_1[reference_col], columns=embeddings_df_2[external_col])
    
#     # Find top matches and their scores
#     top_matches_with_score_df = distance_matrix_df.apply(lambda row: pd.Series([row.idxmax(), row.max()], index=[external_col, 'similarity_score']), axis=1)
    
#     # Save top matches and distance matrix to parquet
#     top_matches_with_score_df.to_parquet(Paths('FNDDS', 'MEXT2015').top_match_path + 'top_match_in_sr_legacy_space.parquet')
#     distance_matrix_df.to_parquet(Paths('FNDDS', 'MEXT2015').distance_matrix + 'distance_matrix_in_sr_legacy_space.parquet')

#     return top_matches_with_score_df, distance_matrix_df


# def embedddings_df():
#     output_path= Paths(reference_database).food_items_path
#     df = pd.read_parquet(output_path)
#     df = df.replace('－', 'Algae')
#     reference_pydantic_class = reference_pydantic_class

#     df[f'{reference_database}_class'] = reference_pydantic_class.df2fooditems(df)


#     df[f'{reference_database}_dict'] = df[f'{reference_database}_class'].apply(lambda x: x.dict())
#     df= df.reset_index()
#     chunk_size = 100

#     embeddings_path = Paths(reference_database, reference_database).embedding_path
#     food_chunks = [df[i : i + chunk_size] for i in range(0, df.shape[0], chunk_size)]
#     chunk  = df
#     pandarallel.initialize(progress_bar=True, nb_workers=64)
#     chunk[f'{reference_database}_embeddings'] = chunk[f'{reference_database}_dict'].parallel_apply(lambda x: get_embedding_parallel_generalised(x,reference_pydantic_class ))
#     chunk.drop(columns=[f'{reference_database}_class'], inplace=True)
#     chunk.drop(columns=[f'{reference_database}_dict'], inplace=True)



#getting japense names
# path = Paths('MEXT2015').raw_data_path + 'mext_2015_japense_version.xlsx'
# raw_table = pd.read_excel(path )
# food_items = pd.read_parquet(Paths('MEXT2015').food_items_path)

# raw_table.columns = raw_table.iloc[2]
# raw_table = raw_table[3:]
# raw_table.columns = raw_table.columns.str.replace('\n', '', regex=True)
# raw_table = raw_table.rename(columns={'食品番号':'Item No.','食品名（全体）':'japense_food_item', '英名': 'english_food_item'})

# japense_names_df = raw_table[['Item No.', 'japense_food_item', 'english_food_item']] 
# food_items = food_items.merge(japense_names_df, on='Item No.', how='left')
# food_items.to_parquet(Paths('MEXT2015').food_items_path)