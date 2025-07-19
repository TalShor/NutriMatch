from typing import ClassVar

from pydantic import Field, validator

from ...base_classes.base_food_item import FoodItem


class ZameretFoodItem(FoodItem):
    hebrew_name: str = Field(..., description="The name of the food item in hebrew.")
    english_name: str = Field(..., description="The name of the food item in English.")
    unique_food_column: ClassVar[str] = "hebrew_name"

    @validator("hebrew_name", "english_name")
    def check_length(cls, v):
        return FoodItem.check_lengths_base(v)

    def __str__(self):
        cls_str = f"hebrew_name: {self.hebrew_name}."
        cls_str += f"\english_name: {self.english_name}."
        return cls_str

    def simplify(self):
        self.english_name = None
