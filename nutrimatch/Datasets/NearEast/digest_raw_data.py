import pandas as pd
from ...Utils import Paths
# import boto3
# import requests
# from bs4 import BeautifulSoup

def create_useable_data():

    # Raw data path
    raw_data_path = Paths('NearEast').raw_data_path

    raw_data_path += '/near-east-food-items.csv'

    # Read in the raw data
    data = pd.read_csv(raw_data_path)

    # Combine the 'FOOD' & 'VARIANT' columns
    data['description'] = data['FOOD'] + ', ' + data['VARIANT']

    # Drop the 'FOOD' & 'VARIANT' columns
    data.drop(columns=['FOOD', 'VARIANT'], inplace=True)

    # Rename 'CATEGORY' to 'food_category'
    data.rename(columns={'CATEGORY': 'food_category'}, inplace=True)

    # Save fooditems to parquet
    data.to_parquet(Paths('NearEast').food_items_path)

# def extract_data_from_near_east_website():
#     pages = range(0, 23) # The pages we want to extract tables from
#     for page in pages:
#         url = get_url(page)
#         tables = get_tables(url)
#         save_tables(tables, prefix=f"page_{page}_")
#         # print(tables)

#     def get_url(page: int):
#         # Get the int as a two digit string
#         page_str = str(page).zfill(2)
#         return f"https://www.fao.org/4/X6879E/X6879E{page_str}.htm"


#     def get_tables(url: str):
#         # Get all the tables on the webpage
#         headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
#         response = requests.get(url, headers=headers)
#         soup = BeautifulSoup(response.text, 'html.parser')
#         try:
#             tables = pd.read_html(str(soup))
#         except ValueError:
#             tables = []
#         return tables

#     def save_tables(tables: list, prefix: str = ""):
#         # Save the tables to a csv
#         for i, table in enumerate(tables):
#             table.to_csv(f"s3://ds-users/shahar/near-east-extraction/tables/{prefix}table_{i}.csv", index=False)

# def reformat_data():
#     # Iterate each extrected table and reformat it

#     tables_path = "shahar/near-east-extraction/tables/"

#     output_dir = "s3://ds-users/shahar/near-east-extraction/reformatted-tables/"

#     s3 = boto3.resource('s3')
#     bucket = s3.Bucket('ds-users')
#     files = list(bucket.objects.filter(Prefix=tables_path))

#     def reformat_food_and_variant(df: pd.DataFrame) -> pd.DataFrame:
#         # Check if the df has the required columns
#         if "NO" in df.columns and "FOOD" in df.columns:
#         # Identify the rows that contain food names and variants
#             food_name = ""
#             variant = ""
#             new_rows = []

#             for index, row in df.iterrows():
#                 if pd.notnull(row['FOOD']) and not row['FOOD'].startswith('-'):
#                     food_name = row['FOOD']
#                 elif pd.notnull(row['FOOD']) and row['FOOD'].startswith('-'):
#                     variant = row['FOOD']
#                     new_row = row.copy()
#                     new_row['FOOD'] = food_name
#                     new_row['VARIANT'] = variant
#                     new_rows.append(new_row)

#             # Create a new DataFrame with the separated columns
#             return pd.DataFrame(new_rows)
#         raise ValueError("The DataFrame does not contain the required columns")

#     for file in files:
#         if file.key.endswith(".csv"):
#             print(file.key)
#             file_name = file.key.split("/")[-1]
#             # Load the df
#             df = pd.read_csv(f"s3://ds-users/{file.key}")
#             # Check the df format
#             if "NO" in df.columns and "FOOD" in df.columns:
#                 # Reformat the df
#                 new_df = reformat_food_and_variant(df)
#                 # Save the df
#                 new_df.to_csv(f"{output_dir}{file_name}", index=False)

# def generate_list_of_food_items():
#     tables_path = "shahar/near-east-extraction/reformatted-tables/"

#     s3 = boto3.resource('s3')
#     bucket = s3.Bucket('ds-users')
#     files = list(bucket.objects.filter(Prefix=tables_path))

#     def get_food_names():
#         include = ["NO", "FOOD", "VARIANT"]
#         food_names = set()
#         for file in files:
#             if file.key.endswith(".csv"):
#                 # Load the df
#                 df = pd.read_csv(f"s3://ds-users/{file.key}")
#                 # Check the df format
#                 if all(col in df.columns for col in include):
#                     # create a set of tuples with the food name and variant
#                     food_names.update(set(zip(df["NO"], df["FOOD"], [variant.replace("-", "") for variant in df["VARIANT"]])))
#         return food_names

#     food_names = get_food_names()
#     # save the food names to a csv
#     food_names_df = pd.DataFrame(food_names, columns=["NO", "FOOD", "VARIANT"])
#     food_names_df.to_csv("s3://ds-users/shahar/near-east-extraction/food-names.csv", index=False)