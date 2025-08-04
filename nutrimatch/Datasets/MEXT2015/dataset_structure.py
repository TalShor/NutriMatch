from pydantic import BaseModel, Field, ValidationError, constr, validator
from typing import Optional, ClassVar
import pandas as pd
import numpy as np
from enum import Enum

from ...Utils.base_food_item import FoodItem


class MEXT2015_diet_categories(Enum):
    Grains = 'Grains'
    Fish_and_Shellfish = 'Fish and Shellfish'
    Meat = 'Meat'
    Eggs = 'Eggs'
    Dairy_Products = 'Dairy Products'
    Fats_and_Oils = 'Fats and Oils'
    Vegetables = 'Vegetables'
    Fruit = 'Fruit'
    Nuts_and_Seeds = 'Nuts and Seeds'
    Sugars_and_Sweeteners = 'Sugars and Sweeteners'
    Seasonings_and_Spices = 'Seasonings and Spices'
    Prepared_and_Processed_Foods = 'Prepared and Processed Foods'
    Beverages = 'Beverages'
    Confectioneries = 'Confectioneries'
    Mushrooms = 'Mushrooms'
    Algae = 'Algae'
    Potatoes_and_Starches = 'Potatoes and Starches'
    Pulses = 'Pulses'
    
class MEXT2015FoodItem(FoodItem):
    food_item: str = Field(...,
        description="The name of the food item in English'.")
    food_category: MEXT2015_diet_categories = Field(...,
        description="The category of the food item in English.")
    unique_food_column: ClassVar[str] = 'food_item'

    @validator('food_item')
    def check_length(cls, v):
        return FoodItem.check_lengths_base(v)
    
    def __str__(self):
        cls_str = f"Food Item: {self.food_item}."
        cls_str += f"\nCategory: {self.food_category.value}."
        return cls_str
    
    def simplify(self):
        self.food_category = None


