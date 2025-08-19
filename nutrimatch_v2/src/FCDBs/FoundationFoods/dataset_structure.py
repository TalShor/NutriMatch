from enum import Enum
from typing import ClassVar

from pydantic import Field, validator

from ...base_classes.base_food_item import FoodItem


class USDA_foundation_categories(Enum):
    Baked_Products = "Baked Products"
    Beef_Products = "Beef Products"
    Beverages = "Beverages"
    Cereal_Grains_and_Pasta = "Cereal Grains and Pasta"
    Dairy_and_Egg_Products = "Dairy and Egg Products"
    Fats_and_Oils = "Fats and Oils"
    Finfish_and_Shellfish_Products = "Finfish and Shellfish Products"
    Fruits_and_Fruit_Juices = "Fruits and Fruit Juices"
    Lamb_Veal_and_Game_Products = "Lamb, Veal, and Game Products"
    Legumes_and_Legume_Products = "Legumes and Legume Products"
    Nut_and_Seed_Products = "Nut and Seed Products"
    Pork_Products = "Pork Products"
    Poultry_Products = "Poultry Products"
    Restaurant_Foods = "Restaurant Foods"
    Sausages_and_Luncheon_Meats = "Sausages and Luncheon Meats"
    Soups_Sauces_and_Gravies = "Soups, Sauces, and Gravies"
    Spices_and_Herbs = "Spices and Herbs"
    Sweets = "Sweets"
    Vegetables_and_Vegetable_Products = "Vegetables and Vegetable Products"


class FoundationFoodsFoodItem(FoodItem):
    description: str = Field(
        ...,
        description="An English description written in the USDA Foundation Foods style (include preparation method, fat content, qualifiers, etc.) – it does *not* have to be an exact string from the database but should match the CLOSEST equivalent food item familiar in the United States *with the same core ingredients* (e.g. chicken schnitzel → fried chicken). All brand or marketing terms must be omitted.",
    )
    food_category: USDA_foundation_categories = Field(
        ...,
        description="USDA Foundation Foods category of the item. Must match one value from USDA_foundation_categories exactly.",
    )
    unique_food_column: ClassVar[str] = "description"

    @validator(
        "description",
    )
    def check_length(cls, v):
        return FoodItem.check_lengths_base(v)

    def __str__(self):
        cls_str = f"Description: {self.description}."
        cls_str += f"\nCategory: {self.food_category.value}."
        return cls_str

    def simplify(self):
        self.food_category = None
