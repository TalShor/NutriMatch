from enum import Enum
from typing import ClassVar

from pydantic import Field, validator

from ...base_classes.base_food_item import FoodItem


class USDA_diet_categories(Enum):
    Dairy_and_Egg_Products = "Dairy and Egg Products"
    Spices_and_Herbs = "Spices and Herbs"
    Baby_Foods = "Baby Foods"
    Fats_and_Oils = "Fats and Oils"
    Poultry_Products = "Poultry Products"
    Soups_Sauces_and_Gravies = "Soups, Sauces, and Gravies"
    Sausages_and_Luncheon_Meats = "Sausages and Luncheon Meats"
    Breakfast_Cereals = "Breakfast Cereals"
    Fruits_and_Fruit_Juices = "Fruits and Fruit Juices"
    Pork_Products = "Pork Products"
    Vegetables_and_Vegetable_Products = "Vegetables and Vegetable Products"
    Nut_and_Seed_Products = "Nut and Seed Products"
    Beef_Products = "Beef Products"
    Beverages = "Beverages"
    Finfish_and_Shellfish_Products = "Finfish and Shellfish Products"
    Legumes_and_Legume_Products = "Legumes and Legume Products"
    Lamb_Veal_and_Game_Products = "Lamb, Veal, and Game Products"
    Baked_Products = "Baked Products"
    Sweets = "Sweets"
    Cereal_Grains_and_Pasta = "Cereal Grains and Pasta"
    Fast_Foods = "Fast Foods"
    Meals_Entrees_and_Side_Dishes = "Meals, Entrees, and Side Dishes"
    Snacks = "Snacks"
    American_Indian_Alaska_Native_Foods = "American Indian/Alaska Native Foods"
    Restaurant_Foods = "Restaurant Foods"
    Branded_Food_Products_Database = "Branded Food Products Database"
    Quality_Control_Materials = "Quality Control Materials"
    Alcoholic_Beverages = "Alcoholic Beverages"
    Dietary_Supplements = "Dietary Supplements"  # this was not originally in the dataset - but HPP had mixed them together so for ease of use we've added it.


class SR_LegacyFoodItem(FoodItem):
    description: str = Field(
        ...,
        description="The name of the food item in English. Remove brand names and other non-descriptive words. e.g. 'Coca Cola' should be 'Cola'.",
    )
    food_category: USDA_diet_categories = Field(
        ..., description="The category of the food item in English."
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
