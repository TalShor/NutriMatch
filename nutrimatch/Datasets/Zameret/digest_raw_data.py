import numpy as np
import pandas as pd
from ...Utils import Paths
# import get_embedding_parallel_generalised
# from dataset_structure import ZameretFoodItem
# from pandarallel import pandarallel
# pandarallel.initialize(progress_bar=True, nb_workers=64)
# from sklearn.metrics.pairwise import cosine_similarity



# TODO: this doesn't do anything.
#creating food items
def create_food_items():
    raw_data = Paths('Zameret').raw_data_path
    df = pd.read_csv(raw_data + '/moh_mitzrachim.csv')
    df = df.rename(columns = {'shmmitzrach': 'hebrew_name' })
    cols_first = ['hebrew_name', 'english_name']
    cols = cols_first + [col for col in df.columns if col not in cols_first]
    df = df[cols]



# TODO: getting embeddings shouldn't be here - it's generic.

# def get_embeddings():
#     output_path= Paths('Zameret').food_items_path
#     df = pd.read_parquet(output_path)
#     #add columns representing the pydantic class for reference database
#     reference_database = 'Zameret'
#     reference_pydantic_class = ZameretFoodItem
#     df['english_name'] = df['english_name'].fillna('No translation')
#     df[f'{reference_database}_class'] = reference_pydantic_class.df2fooditems(df)

#     df[f'{reference_database}_dict'] = df[f'{reference_database}_class'].apply(lambda x: x.dict())
#     df= df.reset_index()
#     chunk_size = 100
#     embeddings_path = Paths('Zameret', 'Zameret').embedding_path
#     food_chunks = [df[i : i + chunk_size] for i in range(0, df.shape[0], chunk_size)]

#     pandarallel.initialize(progress_bar=True, nb_workers=64)
#     for chunk in  food_chunks[46:47]:
#         chunk_number = int(chunk.index[0]/chunk_size)
#         print(chunk_number)
#         chunk['Zameret_embeddings'] = chunk[f'{reference_database}_dict'].parallel_apply(lambda x: get_embedding_parallel_generalised(x,ZameretFoodItem ))
#         chunk.drop(columns=[f'{reference_database}_class'], inplace=True)
#         chunk.to_parquet(f'{embeddings_path}chunk_{chunk_number}.parquet')


# getting matches between Zameret and HPP
def similarities_between_Zameret_HPP():
    hpp_embeddings_path = Paths('SR_Legacy', 'HPP').embedding_path
    Zameret_embeddings_path = Paths('SR_Legacy', 'Zameret').embedding_path
    hpp_embeddings = pd.read_parquet(hpp_embeddings_path)
    Zameret_embeddings = pd.read_parquet(Zameret_embeddings_path , columns = ['hebrew_name', 'english_name', 'Zameret_to_SR_Legacy_embedding'])

    hpp_embeddings = hpp_embeddings.dropna(subset = 'hpp_sr_legacy_embedding')
    chunk_embeddings = np.stack(hpp_embeddings['hpp_sr_legacy_embedding'])
    external_database_embeddings = np.stack(Zameret_embeddings['Zameret_to_SR_Legacy_embedding'])

    similarity_matrix = cosine_similarity(chunk_embeddings, external_database_embeddings )

    # Create DataFrame from the similarity matrix
    distance_matrix_df = pd.DataFrame(similarity_matrix, index=hpp_embeddings['hpp__hebrew_name'], columns=Zameret_embeddings ['hebrew_name'])

    # Find the top match and its score for each display_name
    top_matches_with_score_df = distance_matrix_df.apply(lambda row: pd.Series([row.idxmax(), row.max()], index=['hebrew_name', 'similarity_score']), axis=1)

    top_matches_with_score_df.to_parquet(Paths('Zameret', 'HPP').top_match_path +'top_matches_SR_Legacy_space.parquet')
    distance_matrix_df.to_parquet(Paths('Zameret', 'HPP').distance_matrix_path +'distance_matrix.parquet')
    return top_matches_with_score_df, distance_matrix_df


def get_nutrients():
    zameret_paths = Paths('Zameret')
    nutrient_levels = pd.read_csv(zameret_paths.raw_data_path + '/moh_mitzrachim.csv')
    sr_legacy_nutrient_translation = {
        'protein': 'Protein',
        'total_fat': 'Total lipid (fat)',
        'carbohydrates': 'Carbohydrate, by difference',
        'food_energy': 'Energy',
        'alcohol': 'Alcohol, ethyl',
        'moisture': 'Water',
        'total_dietary_fiber': 'Fiber, total dietary',
        'calcium': 'Calcium, Ca',
        'iron': 'Iron, Fe',
        'magnesium': 'Magnesium, Mg',
        'phosphorus': 'Phosphorus, P',
        'potassium': 'Potassium, K',
        'sodium': 'Sodium, Na',
        'zinc': 'Zinc, Zn',
        'copper': 'Copper, Cu',
        'vitamin_a_iu': 'Vitamin A, IU',
        'carotene': 'Carotene, beta',
        'vitamin_e': 'Vitamin E (alpha-tocopherol)',
        'vitamin_c': 'Vitamin C, total ascorbic acid',
        'thiamin': 'Thiamin',
        'riboflavin': 'Riboflavin',
        'niacin': 'Niacin',
        'vitamin_b6': 'Vitamin B-6',
        'folate': 'Folate, total',
        'folate_dfe': 'Folate, DFE',
        'vitamin_b12': 'Vitamin B-12',
        'cholesterol': 'Cholesterol',
        'saturated_fat': 'Fatty acids, total saturated',
        'butyric': 'SFA 4:0',  # assuming butyric acid
        'caproic': 'SFA 6:0',  # assuming caproic acid
        'caprylic': 'SFA 8:0',  # assuming caprylic acid
        'capric': 'SFA 10:0',  # assuming capric acid
        'lauric': 'SFA 12:0',  # assuming lauric acid
        'myristic': 'SFA 14:0',  # assuming myristic acid
        'palmitic': 'SFA 16:0',  # assuming palmitic acid
        'stearic': 'SFA 18:0',  # assuming stearic acid
        'oleic': 'MUFA 18:1 c',  # assuming oleic acid
        'linoleic': 'PUFA 18:2 n-6 c,c',  # assuming linoleic acid
        'linolenic': 'PUFA 18:3 n-3 c,c,c (ALA)',  # assuming alpha-linolenic acid
        'arachidonic': 'PUFA 20:4 n-6',  # assuming arachidonic acid
        'docosahexanoic': 'PUFA 22:6 n-3 (DHA)',  # assuming docosahexaenoic acid
        'palmitoleic': 'MUFA 16:1',  # assuming palmitoleic acid
        'parinaric': 'PUFA 18:4',  # assuming parinaric acid
        'gadoleic': 'MUFA 20:1',  # assuming gadoleic acid
        'eicosapentaenoic': 'PUFA 20:5 n-3 (EPA)',  # assuming eicosapentaenoic acid
        'erucic': 'MUFA 22:1 c',  # assuming erucic acid
        'docosapentaenoic': 'PUFA 22:5 n-3 (DPA)',  # assuming docosapentaenoic acid
        'mono_unsaturated_fat': 'Fatty acids, total monounsaturated',
        'poly_unsaturated_fat': 'Fatty acids, total polyunsaturated',
        'vitamin_d': 'Vitamin D (D2 + D3)',
        'total_sugars': 'Sugars, Total',
        'trans_fatty_acids': 'Fatty acids, total trans',
        'vitamin_a_re': 'Vitamin A, RAE',
        'isoleucine': 'Isoleucine',
        'leucine': 'Leucine',
        'valine': 'Valine',
        'lysine': 'Lysine',
        'threonine': 'Threonine',
        'methionine': 'Methionine',
        'phenylalanine': 'Phenylalanine',
        'tryptophan': 'Tryptophan',
        'histidine': 'Histidine',
        'tyrosine': 'Tyrosine',
        'arginine': 'Arginine',
        'cystine': 'Cystine',
        'serine': 'Serine',
        'pantothenic_acid': 'Pantothenic acid',
        'selenium': 'Selenium, Se',
        'choline': 'Choline, total',
        'manganese': 'Manganese, Mn',
        'fructose': 'Fructose'
    }

    unique_vars = {'vitamin_k': 'Vitamin K',
    'iodine': 'Iodine',
    'sugar_alcohols': 'Sugar alcohols',
    'biotin': 'Biotin'}

    nutrient_levels\
        .rename(columns=sr_legacy_nutrient_translation)\
        .rename(columns=unique_vars)\
        .set_index('shmmitzrach')[list(sr_legacy_nutrient_translation.values()) + list(unique_vars.values())]\
        .to_parquet(zameret_paths.nutrients_path)


if __name__ == "__main__":
    get_nutrients()