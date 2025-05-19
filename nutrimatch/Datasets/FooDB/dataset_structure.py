from pydantic import BaseModel, Field, ValidationError, constr, validator
from typing import Optional, Type, List, Literal, ClassVar
from enum import Enum
from ...Utils.base_food_item import FoodItem


class FoodDB_food_group(Enum):
    Herbs_and_Spices = 'Herbs and Spices'
    Vegetables = 'Vegetables'
    Fruits = 'Fruits'
    Nuts = 'Nuts'
    Cereals_and_Cereal_Products = 'Cereals and cereal products'
    Pulses = 'Pulses'
    Teas = 'Teas'
    Gourds = 'Gourds'
    Coffee_and_Coffee_Products = 'Coffee and coffee products'
    Soy = 'Soy'
    Cocoa_and_Cocoa_Products = 'Cocoa and cocoa products'
    Beverages = 'Beverages'
    Aquatic_Foods = 'Aquatic foods'
    Animal_Foods = 'Animal foods'
    Milk_and_Milk_Products = 'Milk and milk products'
    Eggs = 'Eggs'
    Confectioneries = 'Confectioneries'
    Baking_Goods = 'Baking goods'
    Dishes = 'Dishes'
    Snack_Foods = 'Snack foods'
    Baby_Foods = 'Baby foods'
    Unclassified = 'Unclassified'
    Fats_and_Oils = 'Fats and oils'
    Herbs_and_Spices_2 = 'Herbs and spices'
    Other_seeds = 'Other seeds'


class FoodDB_food_subgroup(Enum):
    Herbs = 'Herbs'
    Cabbages = 'Cabbages'
    Tropical_Fruits = 'Tropical fruits'
    Onion_Family_Vegetables = 'Onion-family vegetables'
    Nuts = 'Nuts'
    Spices = 'Spices'
    Root_Vegetables = 'Root vegetables'
    Shoot_Vegetables = 'Shoot vegetables'
    Cereals = 'Cereals'
    Leaf_Vegetables = 'Leaf vegetables'
    Oilseed_Crops = 'Oilseed crops'
    Peas = 'Peas'
    Teas = 'Teas'
    Fruit_Vegetables = 'Fruit vegetables'
    Gourds = 'Gourds'
    Citrus = 'Citrus'
    Coffee = 'Coffee'
    Pomes = 'Pomes'
    Berries = 'Berries'
    Other_Fruits = 'Other fruits'
    Soy = 'Soy'
    Tubers = 'Tubers'
    Lentils = 'Lentils'
    Other_Pulses = 'Other pulses'
    Beans = 'Beans'
    Drupes = 'Drupes'
    Stalk_Vegetables = 'Stalk vegetables'
    Cocoa = 'Cocoa'
    Fermented_Beverages = 'Fermented beverages'
    Other_Breads = 'Other breads'
    Cereal_Products = 'Cereal products'
    Soy_Products = 'Soy products'
    Doughs = 'Doughs'
    Distilled_Beverages = 'Distilled beverages'
    Fortified_Wines = 'Fortified wines'
    Alcoholic_Beverages = 'Alcoholic beverages'
    Mollusks = 'Mollusks'
    Seaweed = 'Seaweed'
    Crustaceans = 'Crustaceans'
    Fishes = 'Fishes'
    Cetaceans = 'Cetaceans'
    Bovines = 'Bovines'
    Swine = 'Swine'
    Other_Seeds = 'Other seeds'
    Other_Vegetables = 'Other vegetables'
    Poultry = 'Poultry'
    Venison = 'Venison'
    Equines = 'Equines'
    Other_Aquatic_Foods = 'Other aquatic foods'
    Pinnipeds = 'Pinnipeds'
    Lagomorphs = 'Lagomorphs'
    Ovis = 'Ovis'
    Caprae = 'Caprae'
    Mushrooms = 'Mushrooms'
    Amphibians = 'Amphibians'
    Fermented_Milk_Products = 'Fermented milk products'
    Unfermented_Milks = 'Unfermented milks'
    Eggs = 'Eggs'
    Frozen_Desserts = 'Frozen desserts'
    Other_Confectioneries = 'Other confectioneries'
    Candies = 'Candies'
    Seasonings = 'Seasonings'
    Desserts = 'Desserts'
    Other_Dishes = 'Other dishes'
    Snack_Foods = 'Snack foods'
    Flat_Breads = 'Flat breads'
    Dressings = 'Dressings'
    Sauces = 'Sauces'
    Other_Milk_Products = 'Other milk products'
    Substitutes = 'Substitutes'
    Sugars = 'Sugars'
    Ground_Meat = 'Ground meat'
    Condiments = 'Condiments'
    Baking_Goods = 'Baking goods'
    Fruit_Products = 'Fruit products'
    Waters = 'Waters'
    Fish_Products = 'Fish products'
    Other_Beverages = 'Other beverages'
    Baby_Foods = 'Baby foods'
    Vegetable_Products = 'Vegetable products'
    Unclassified = 'Unclassified'
    Animal_Fats = 'Animal fats'
    Spreads = 'Spreads'
    Herb_and_Spice_Mixtures = 'Herb and spice mixtures'
    Cocoa_Products = 'Cocoa products'
    Fermented_Milks = 'Fermented milks'
    Leavened_Breads = 'Leavened breads'
    Roe = 'Roe'
    Nutritional_Beverages = 'Nutritional beverages'
    Tex_Mex_Cuisine = 'Tex-Mex cuisine'
    Sandwiches = 'Sandwiches'
    Milk_Desserts = 'Milk desserts'
    Asian_Cuisine = 'Asian cuisine'
    Herbal_Teas = 'Herbal teas'
    Pasta_Dishes = 'Pasta dishes'
    Berber_Cuisine = 'Berber cuisine'
    Coffee_Products = 'Coffee products'
    Mexican_Cuisine = 'Mexican cuisine'
    Potato_Dishes = 'Potato dishes'
    American_Cuisine = 'American cuisine'
    Wrappers = 'Wrappers'
    Vegetable_Fats = 'Vegetable fats'
    Latin_American_Cuisine = 'Latin American cuisine'
    Bread_Products = 'Bread products'
    Sweet_Breads = 'Sweet breads'
    Jewish_Cuisine = 'Jewish cuisine'
    Levantine_Cuisine = 'Levantine cuisine'
    Brassicas = 'Brassicas'
    Cereals_and_Cereal_Products = 'Cereals and cereal products'
    Cocoa_and_Cocoa_Products = 'Cocoa and cocoa products'
    Coffee_and_Coffee_Products = 'Coffee and coffee products'
    Milk_and_Milk_Products = 'Milk and milk products'
    Fats_and_Oils = 'Fats and oils'
    Herbs_and_Spices = 'Herbs and Spices'
    Pulses = 'Pulses'
    Beverages = 'Beverages'
    Fruits = 'Fruits'
    Green_Vegetables = 'Green vegetables'
    Bivalvia = 'Bivalvia'



class FooDBFoodItem(FoodItem):
    name: str = Field(...,
        description="The name of the food item in English. Keep only the translated name in english. e.g. 'אבוקדו' should be 'Avocado'.")
    food_group:  FoodDB_food_group = Field(..., 
        description="The category of the food item in English. If a match can't be found classify it as 'Unclassified'.")
    food_subgroup: FoodDB_food_subgroup = Field(...,
        description="The subcategory of the food item in English. If a match can't be found classify it as 'Unclassified'")
    unique_food_column: ClassVar[str] = 'name'
    #TODO make sure one are enum are correlated with food groups and subgroups
    @validator('name')
    def check_length(cls, v):
        return FoodItem.check_lengths_base(v)
    
    @validator('food_group', pre=True)
    def ensure_valid_group(cls, v):
        if v not in FoodDB_food_group._value2member_map_:
            return FoodDB_food_group.Unclassified
        return v

    @validator('food_subgroup', pre=True)
    def ensure_valid_subgroup(cls, v):
        if v not in FoodDB_food_subgroup._value2member_map_:
            return FoodDB_food_subgroup.Unclassified
        return v

    def __str__(self):
        cls_str = f"Name: {self.name}."
        cls_str += f"\nFood_group: {self.food_group.value}."
        cls_str += f"\nFood_sub_group: {self.food_subgroup.value}."
        return cls_str
    
    def simplify(self):
        self.food_group = None
        self.food_subgroup = None
