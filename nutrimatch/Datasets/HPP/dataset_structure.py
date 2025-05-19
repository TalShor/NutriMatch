from pydantic import BaseModel, Field, ValidationError, constr, validator
from typing import Optional, Type, List, Literal, ClassVar, Optional
import pandas as pd
import numpy as np
from enum import Enum
from ...Utils.base_food_item import FoodItem

class HPPFoodItem(FoodItem):
    hebrew_name: str = Field(..., 
        description="The name of the food item in Hebrew.")
    short_description: Optional[str] = Field(None, 
        description="A short description of the food item in English.")
    category_hint: Optional[str] = Field(None, 
        description="A hint to the food item's category.")
    
    unique_food_column: ClassVar[str] = 'hebrew_name'

    # debug_this:bool = True
    # I removed 'hebrew_name' because of bugs in the names...
    @validator('short_description', 'category_hint')
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



class HPPServingUnits(Enum):
    Gram = 'gram'
    Very_Small_Unit = 'very small unit'
    Large_Bottle = 'large bottle'
    Small_Portion = 'small portion'
    Long = 'long'
    Italian_Tomato = 'Italian tomato'
    Small = 'small'
    Cup_Chopped_Or_Sliced = 'cup chopped or sliced'
    Bottle = 'bottle'
    Cherry_Tomato = 'cherry tomato'
    Cup = 'cup'
    Large_Carton = 'large carton'
    Tablespoon = 'tablespoon'
    Medium = 'medium'
    Small_Unit = 'small unit'
    Cup_Of_Cherry_Tomatoes = 'cup of cherry tomatoes'
    Large_Portion = 'large portion'
    Grams = 'grams'
    Cup_Mashed = 'cup mashed'
    Carton = 'carton'
    Medium_Portion = 'medium portion'
    Thin_Slice = 'thin slice'
    Slice = 'slice'
    Large_Unit = 'large unit'
    Medium_Unit = 'medium unit'
    Teaspoon = 'teaspoon'
    Half_Cup_Slices = 'half cup slices'
    Kilogram = 'kilogram'
    Small_Carton = 'small carton'
    Bag = 'bag'
    not_relevant = 'not_relevant'
    # Very_Small_Cherry_Unit = 'very small cherry unit small'





  