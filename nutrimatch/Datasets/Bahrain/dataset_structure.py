from ...Utils.base_food_item import FoodItem
from pydantic import Field, validator
from enum import Enum
from typing import ClassVar

class Bahrain_diet_categories(Enum):
    Cereal_and_Cereal_Products = 'Cereal and Cereal Products'
    Bread_and_Bread_Products = 'Bread and Bread Products'
    Fruit = 'Fruit'
    Vegetables = 'Vegetables'
    Legumes = 'Legumes'
    Nuts_and_Seeds = 'Nuts and Seeds'
    Meat_Poultry_and_Egg = 'Meat Poultry and Egg'
    Fish = 'Fish'
    Milk_and_Dairy_Products = 'Milk and Dairy Products'
    Fats_and_Oils = 'Fats and Oils'
    Herbs_and_Spices = 'Herbs and Spices'
    Beverages = 'Beverages'
    Local_Fast_Food = 'Local Fast Food'
    Western_Fast_Food = 'Western Fast Food'
    Miscellaneous = 'Miscellaneous'
    Ready_To_Eat_Food = 'Ready To Eat Food'
    Bakery_Products = 'Bakery Products'
    Local_and_Western_Fast_Food = 'Local and Western Fast Food'
    Traditional_Confections = 'Traditional Confections'
    Pizzas = 'Pizzas'
    Raw_and_Cooked_Fish = 'Raw and Cooked Fish'
    Unclassified = 'Unclassified'

class Bahrain_diet_subcategories(Enum):
    Fast_Foods_of_Local_Origin = 'Fast foods of local origin'
    Fast_Foods_of_Western_Origin = 'Fast foods of western origin'
    Raw = 'Raw'
    Grilled = 'Grilled'
    Curried = 'Curried'
    Fried = 'Fried'
    Cooked_in_Rice = 'Cooked in rice'
    Unclassified = 'Unclassified'

class BahrainFoodItem(FoodItem):
    food: str = Field(...,
        description="The name of the food item in English. Keep only the translated name in english. e.g. 'אבוקדו' should be 'Avocado'.")

    food_category: Bahrain_diet_categories = Field(...,
        description="The category of the food item in English. If a match can't be found classify it as 'Unclassified'.")
    
    food_subcategory: Bahrain_diet_subcategories = Field(...,
        description="The subcategory of the food item in English. If a match can't be found classify it as 'Unclassified'.")

    local_name: str|None = Field(...,
        description="The local name of the food item in Bahrain in English.")
    
    scientific_name: str|None = Field(...,
        description="The scientific name of the food item in English.")
    
    arabic_name: str|None = Field(...,
        description="The name of the food item in Arabic.")

    unique_food_column: ClassVar[str] = 'food'

    @validator('food')
    def check_length(cls, v):
        return FoodItem.check_lengths_base(v)

    @validator('food_category', pre=True)
    def ensure_valid_category(cls, v):
        if v not in Bahrain_diet_categories._value2member_map_:
            return Bahrain_diet_categories.Unclassified
        return v

    @validator('food_subcategory', pre=True)
    def ensure_valid_subcategory(cls, v):
        if v not in Bahrain_diet_subcategories._value2member_map_:
            return Bahrain_diet_subcategories.Unclassified
        return v
    
    def __str__(self):
        cls_str = f"Food: {self.food}."
        cls_str += f"\nFood_category: {self.food_category.value}."
        cls_str += f"\nFood_subcategory: {self.food_subcategory.value}."
        cls_str += f"\nLocal_name: {self.local_name}."
        cls_str += f"\nScientific_name: {self.scientific_name}."
        cls_str += f"\nArabic_name: {self.arabic_name}."
        return cls_str
    
    def simplify(self):
        self.food_category = None
        self.food_subcategory = None
        self.local_name = None
        self.scientific_name = None
        self.arabic_name = None