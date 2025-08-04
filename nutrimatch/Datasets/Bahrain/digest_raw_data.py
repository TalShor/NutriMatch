from ...Utils import Paths
import pandas as pd

def create_useable_format():
    # Load the Bahrain raw data
    raw_data_path = Paths("Bahrain").raw_data_path

    # Attach the file name to the path
    raw_data_path = raw_data_path + "/bahrain_food_registry_with_arabic_name.xlsx"

    # Load the raw data
    data = pd.read_excel(raw_data_path)

    # Replace the empty categories and subcategories with 'Unclassified'
    data.food_category = data.food_category.replace(pd.NA, 'Unclassified')
    data.food_subcategory = data.food_subcategory.replace(pd.NA, 'Unclassified')

    # Drop all but relevant columns
    data = data[['food', 'food_category', 'food_subcategory', 'local_name', 'scientific_name', 'arabic_name']]

    # Save the data in a parquet file
    data.to_parquet(Paths("Bahrain").food_items_path)


    

