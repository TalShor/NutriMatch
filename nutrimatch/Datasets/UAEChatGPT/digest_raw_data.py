from ...Utils import Paths
import pandas as pd

def organize_food_item_category():
    """
    This function replaces each empty category with 'Uncalssified'
    """
    raw_data_path = Paths('UAEChatGPT').raw_data_path
    data = pd.read_excel(raw_data_path +'/uae-common-foods.xlsx')
    data.category = data.category.replace(pd.NA, 'Unclassified')
    data.to_parquet(Paths("UAEChatGPT").food_items_path)
