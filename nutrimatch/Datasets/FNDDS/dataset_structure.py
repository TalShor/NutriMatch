from pydantic import BaseModel, Field, ValidationError, constr, validator
from typing import Optional, Type, List, Literal, ClassVar
from enum import Enum

from ...Utils.base_food_item import FoodItem
import os

class FNDDS_diet_categories(Enum):
    Dairy_Products = 'Dairy Products'
    Soups_and_Sauces = 'Soups and Sauces'
    Baby_Foods = 'Baby Foods'
    Beverages = 'Beverages'
    Prepared_Foods = 'Prepared Foods'
    Unclassified = 'Unclassified'
    Desserts_and_Sweets = 'Desserts and Sweets'
    Mixed_Dishes = 'Mixed Dishes'
    Eggs_and_Egg_Dishes = 'Eggs and Egg Dishes'
    Whole_Meats = 'Whole Meats'
    Fast_Food_Items = 'Fast Food Items'
    Seafood = 'Seafood'
    Vegetables = 'Vegetables'
    Legumes_and_Beans = 'Legumes and Beans'
    Snack_Foods = 'Snack Foods'
    Nuts_and_Seeds = 'Nuts and Seeds'
    Breakfast_Items = 'Breakfast Items'
    Fruits_and_Juices = 'Fruits and Juices'
    Baked_Goods = 'Baked Goods'
    Grain_Based_Snacks = 'Grain-Based Snacks'
    Fried_Foods = 'Fried Foods'
    Fats_and_Oils = 'Fats and Oils'

class FNDDSFoodItem(FoodItem):
    description: str = Field(...,
        description="The name of the food item in English. Keep only the translated name in english. e.g. 'אבוקדו' should be 'Avocado'.")
    food_category:  FNDDS_diet_categories = Field(..., 
        description="The category of the food item in English. If a match can't be found classify it as 'Unclassified'.")
    food_subcategory:  str = Field(..., 
        description="The shorthand English designation of the food item, focusing on its primary characteristic. For example, 'Milk' for 'Milk 3%' and 'Soup' for 'Tomato Soup'. Use this field for a concise yet descriptive label that categorizes similar items under a common term'")
    unique_food_column: ClassVar[str] = 'description'

    @validator('description')
    def check_length(cls, v):
        return FoodItem.check_lengths_base(v)
    
    @validator('food_category', pre=True)
    def ensure_valid_group(cls, v):
        if v not in FNDDS_diet_categories._value2member_map_:
            return FNDDS_diet_categories.Unclassified
        return v

    def __str__(self):
        cls_str = f"Description: {self.description}."
        cls_str += f"\nfood_category: {self.food_category.value}."
        cls_str += f"\nfood_sub_category: {self.food_subcategory}."
        return cls_str
    
    def simplify(self):
        self.food_category = None