from ...Utils.base_food_item import FoodItem
from pydantic import Field, validator
from enum import Enum
from typing import ClassVar

class NearEast_diet_categories(Enum):
    Cereals_and_Grain_Products = 'CEREALS AND GRAIN PRODUCTS'
    Starchy_Roots_and_Tubers = 'STARCHY ROOTS AND TUBERS'
    Dry_Grain_Legumes_and_Legume_Products = 'DRY GRAIN LEGUMES AND LEGUME PRODUCTS'
    Nuts_and_Seeds = 'NUTS AND SEEDS'
    Vegetables = 'VEGETABLES'
    Fruits = 'FRUITS'
    Sugars_Strups_and_Sweets = 'SUGARS, STRUPS AND SWEETS'
    Meat_and_Poultry = 'MEAT AND POULTRY'
    Eggs = 'EGGS'
    Fish_and_Shellfish = 'FISH AND SHELLFISH'
    Milk_and_Milk_Products = 'MILK AND MILK PRODUCTS'
    Oils_and_Fats = 'OILS AND FATS'
    Beverages = 'BEVERAGES'
    Miscellaneous = 'MISCELLANEOUS'
    Unclassified = 'Unclassified'

class NearEastFoodItem(FoodItem):
    description: str = Field(...,
        description="The name of the food item in English.")
    
    food_category: NearEast_diet_categories = Field(...,
        description="The category of the food item in English. If a match can't be found classify it as 'Unclassified'.")

    unique_food_column: ClassVar[str] = 'description'

    @validator('description')
    def check_length(cls, v):
        return FoodItem.check_lengths_base(v)

    @validator('food_category', pre=True)
    def ensure_valid_subgroup(cls, v):
        if v not in NearEast_diet_categories._value2member_map_:
            return NearEast_diet_categories.Unclassified
        return v

    def __str__(self):
        cls_str = f"Description: {self.description}."
        cls_str += f"\nfood_category: {self.food_category.value}."
        return cls_str
    
    def simplify(self):
        self.food_category = None
    
