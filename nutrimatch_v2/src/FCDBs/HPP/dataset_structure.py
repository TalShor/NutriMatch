from typing import ClassVar, Optional

from pydantic import Field, validator

from ...base_classes.base_food_item import FoodItem


class HPPFoodItem(FoodItem):
    hebrew_name: str = Field(..., description="The name of the food item in Hebrew.")
    short_description: Optional[str] = Field(
        None, description="A short description of the food item in English."
    )
    category_hint: Optional[str] = Field(
        None, description="A hint to the food item's category."
    )

    unique_food_column: ClassVar[str] = "hebrew_name"

    # debug_this:bool = True
    # I removed 'hebrew_name' because of bugs in the names...
    @validator("short_description", "category_hint")
    def check_length(cls, v):
        # TODO: maybe use classmethod
        return FoodItem.check_lengths_base(v)

    def __str__(self):
        cls_str = f"Name in Hebrew: {self.hebrew_name}."
        if self.short_description is not None:
            cls_str += f"\nShort Description: {self.short_description}."
        if self.category_hint is not None:
            cls_str += f"\nCategory Hint: {self.category_hint}."
        return cls_str

    def simplify(self):
        self.short_description = None
        self.category_hint = None
