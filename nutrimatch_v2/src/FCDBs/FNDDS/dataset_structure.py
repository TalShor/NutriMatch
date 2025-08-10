from enum import Enum
from typing import ClassVar

from pydantic import Field, validator

from ...base_classes.base_food_item import FoodItem


class FNDDS_diet_categories(Enum):
    FRUIT = "FRUIT"
    VEGETABLES = "VEGETABLES"
    BEVERAGES = "BEVERAGES"
    ALCOHOLIC_BEVERAGES = "ALCOHOLIC BEVERAGES"
    WATER = "WATER"
    FATS_AND_OILS = "FATS AND OILS"
    CONDIMENTS_AND_SAUCES = "CONDIMENTS AND SAUCES"
    SUGARS = "SUGARS"
    BABY_FOODS_AND_FORMULAS = "BABY FOODS AND FORMULAS"
    OTHER = "OTHER"
    MILK_AND_DAIRY = "MILK AND DAIRY"
    PROTEIN_FOODS = "PROTEIN FOODS"
    MIXED_DISHES = "MIXED DISHES"
    GRAINS = "GRAINS"
    SNACKS_AND_SWEETS = "SNACKS AND SWEETS"


class FNDDSFoodItem(FoodItem):
    description: str = Field(
        ...,
        description="The name of the food item in English. Keep only the translated name in english. e.g. 'אבוקדו' should be 'Avocado'.",
    )
    food_category: FNDDS_diet_categories = Field(
        ...,
        description="The category of the food item in English. If a match can't be found classify it as 'Unclassified'.",
    )
    food_subcategory: str = Field(
        ...,
        description="The shorthand English designation of the food item, focusing on its primary characteristic. For example, 'Milk' for 'Milk 3%' and 'Soup' for 'Tomato Soup'. Use this field for a concise yet descriptive label that categorizes similar items under a common term'",
    )
    unique_food_column: ClassVar[str] = "description"

    @validator("description")
    def check_length(cls, v):
        return FoodItem.check_lengths_base(v)

    @validator("food_category", pre=True)
    def ensure_valid_group(cls, v):
        if v not in FNDDS_diet_categories._value2member_map_:
            return FNDDS_diet_categories.OTHER
        return v

    def __str__(self):
        cls_str = f"Description: {self.description}."
        cls_str += f"\nfood_category: {self.food_category.value}."
        cls_str += f"\nfood_sub_category: {self.food_subcategory}."
        return cls_str
