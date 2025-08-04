from enum import Enum
from pydantic import Field, validator
from ...Utils.base_food_item import FoodItem
from typing import ClassVar

class UAEChatGPT_diet_categories(Enum):
    Meat_Poultry_and_Egg = 'MEAT, POULTRY & EGG'
    Milk_and_Dairy_Products = 'MILK & DAIRY PRODUCTS'
    Bread_and_Bread_Products = 'BREAD & BREAD PRODUCTS'
    Fruit = 'FRUIT'
    Vegetables = 'VEGETABLES'
    Legumes = 'LEGUMES'
    Nuts_and_Seeds = 'NUTS & SEEDS'
    Fish = 'FISH'
    Fats_and_Oils = 'FATS & OILS'
    Herbs_and_Spices = 'HERBS & SPICES'
    Beverages = 'BEVERAGES'
    Local_fast_Food = 'LOCAL FAST FOOD'
    Western_fast_Food = 'WESTERN FAST FOOD'
    Unclassified = 'Unclassified'

class UAEChatGPTFoodItem(FoodItem):
    food: str = Field(...,
        description="The name of the food item in English.")
    
    category: UAEChatGPT_diet_categories = Field(...,
        description="The category of the food item in English.")

    unique_food_column: ClassVar[str] = 'food'

    @validator('food')
    def check_length(cls, v):
        return FoodItem.check_lengths_base(v)

    @validator('category', pre=True)
    def ensure_valid_subgroup(cls, v):
        if v not in UAEChatGPT_diet_categories._value2member_map_:
            return UAEChatGPT_diet_categories.Unclassified
        return v

    def __str__(self):
        cls_str = f"Description: {self.food}."
        cls_str += f"\nfood_category: {self.category.value}."
        return cls_str
    
    def simplify(self):
        self.category = None