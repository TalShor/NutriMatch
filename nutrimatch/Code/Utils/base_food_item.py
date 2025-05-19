from pydantic import BaseModel, Field
from typing import Type, List, Literal
import pandas as pd
import numpy as np


class FoodItem(BaseModel):
    unlikely_food_item: bool = Field(default=False, 
        description="A flag to indicate that the food item is unlikely to be a food item. This is useful for filtering out non-food items from the dataset. e.g. dog milk.")

    food_item_discrepancy: Literal['not same food item', 'likely not same food item', 'likely same food item', 'same food item'] = Field(default='same food item',
        description="This field is used to indicate the consistency between different fields of a food item. If all fields (like description, category, etc.) refer to the same food item, the value should be 'same food item'. If there are discrepancies, choose from 'likely not same food item' or 'not same food item'. For example, if the description is 'Avocado' and the category is 'vegetables and fruits', the value should be 'same food item'. But if the description is 'Avocado' and the category is 'oil', the value should be 'not same food item'.")

    def simplify(self):
        raise NotImplementedError

    @classmethod
    def df2fooditems(cls: Type['FoodItem'], df: pd.DataFrame) -> List['FoodItem']:
        # get which attributes we added (all the fields)
        required_attrs = [attr for attr in cls.model_fields]
        required_attrs = np.setdiff1d(required_attrs, list(FoodItem.model_fields.keys()))
        if df.columns.intersection(required_attrs).size != len(required_attrs):
            raise ValueError(f"DataFrame does not contain all the required attributes for {cls}.")
        
        # doesn't work with np.nan
        df.replace({np.nan: None}, inplace=True)
        return df[required_attrs].apply(lambda row: cls(**row.to_dict()), axis=1)
    
    @classmethod
    def fooditems2df(cls, food_items):
        if isinstance(food_items, pd.Series):
            food_items = food_items.tolist()

        # Extract data using the actual class of each item, accommodating for subclass attributes
        data = [{field: getattr(item, field, None) for field in item.__fields__} for item in food_items]

        return pd.DataFrame(data)
    
    def check_lengths_base(str_validation:str) -> str:
        if str_validation is None:
            return str_validation
        
        str_validation = str_validation\
            .replace('\n', '').replace('\t', ' ')\
            .replace('-', ' ').strip().lower()\
            .replace('\\', ' ').replace('/', ' ')\
            .replace('  ', ' ').replace('":', '').replace(',', ' ')
        

        # no words over 20 characters
        if any(len(word) > 20 for word in str_validation.split()):
            print('no words over 20 characters')
            return None

        cleared = str_validation.replace(',', '')\
            .replace('"', '').replace("'", '')\
            .replace('(', '').replace(')', '')\
            .replace('\t', '').replace(' ', '')\
            .replace('":', '').replace('_', '').replace('/', ' ')\
            .strip()
        if len(cleared) < 2 or len(cleared) > 200:
            print (f'description must be between 2 and 200 characters for the word {cleared}')
            return None
          
        
        return str_validation