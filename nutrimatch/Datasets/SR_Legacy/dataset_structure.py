from pydantic import BaseModel, Field, ValidationError, constr, validator
from typing import Optional, ClassVar
import pandas as pd
import numpy as np
from enum import Enum

from ...Utils.base_food_item import FoodItem


class USDA_diet_categories(Enum):
    Dairy_and_Egg_Products = 'Dairy and Egg Products'
    Spices_and_Herbs = 'Spices and Herbs'
    Baby_Foods = 'Baby Foods'
    Fats_and_Oils = 'Fats and Oils'
    Poultry_Products = 'Poultry Products'
    Soups_Sauces_and_Gravies = 'Soups, Sauces, and Gravies'
    Sausages_and_Luncheon_Meats = 'Sausages and Luncheon Meats'
    Breakfast_Cereals = 'Breakfast Cereals'
    Fruits_and_Fruit_Juices = 'Fruits and Fruit Juices'
    Pork_Products = 'Pork Products'
    Vegetables_and_Vegetable_Products = 'Vegetables and Vegetable Products'
    Nut_and_Seed_Products = 'Nut and Seed Products'
    Beef_Products = 'Beef Products'
    Beverages = 'Beverages'
    Finfish_and_Shellfish_Products = 'Finfish and Shellfish Products'
    Legumes_and_Legume_Products = 'Legumes and Legume Products'
    Lamb_Veal_and_Game_Products = 'Lamb, Veal, and Game Products'
    Baked_Products = 'Baked Products'
    Sweets = 'Sweets'
    Cereal_Grains_and_Pasta = 'Cereal Grains and Pasta'
    Fast_Foods = 'Fast Foods'
    Meals_Entrees_and_Side_Dishes = 'Meals, Entrees, and Side Dishes'
    Snacks = 'Snacks'
    American_Indian_Alaska_Native_Foods = 'American Indian/Alaska Native Foods'
    Restaurant_Foods = 'Restaurant Foods'
    Branded_Food_Products_Database = 'Branded Food Products Database'
    Quality_Control_Materials = 'Quality Control Materials'
    Alcoholic_Beverages = 'Alcoholic Beverages'
    Dietary_Supplements = 'Dietary Supplements'

class SR_LegacyFoodItem(FoodItem):
    description: str = Field(..., 
        description="The name of the food item in English. Remove brand names and other non-descriptive words. e.g. 'Coca Cola' should be 'Cola'.")
    food_category: USDA_diet_categories = Field(..., 
        description="The category of the food item in English.")
    common_name: Optional[str] = Field(None,
        description="Common names associated with a food. Could be some commonly used aggragation or just the common name of the food item in English.")
    unique_food_column: ClassVar[str] = 'description'

    @validator('description', 'common_name')
    def check_length(cls, v):
        return FoodItem.check_lengths_base(v)

    def __str__(self):
        cls_str = f"Description: {self.description}."
        cls_str += f"\nCategory: {self.food_category.value}."
        if self.common_name is not None:
            cls_str += f"\nCommon Name: {self.common_name}."
        return cls_str

    def simplify(self):
        self.food_category = None
        self.common_name = None
