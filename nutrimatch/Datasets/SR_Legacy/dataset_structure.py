"""
English naming guidelines for SR_LegacyFoodItem descriptions
----------------------------------------------------------------
These rules are applied by the translation pipeline to ensure
concise, human-readable USDA-style names.

1. Capitalisation & proper nouns – Capitalise the first word and every
   proper noun (Ben & Jerry’s, Israeli). Generic food words remain
   lowercase unless they start the name.
2. Word order – Base food first, then key qualifier(s), then extras,
   e.g. "Cheddar cheese (cow’s milk), 32 % fat, natural".
3. Clarifications – Put non-essential clarifications in parentheses.
4. Lists – Separate attributes with commas; use "and" only before the
   last item.
5. Cooking state – Goes last, preferably in parentheses, e.g.
   "Beef goulash, cooked (meat only)", or "(raw)" if uncooked.
6. Translation artefacts – Replace phrases like "ns as to …" with a
   clear English note in parentheses ("part not specified"). Remove
   placeholders such as "water added 10 %" unless nutritionally
   important; otherwise phrase clearly.
7. Culinary terminology – Prefer standard terms ("whole milk", "sirloin")
   over literal translations.
8. Numbers & units – No space before %, write "3 % fat"; vitamins in
   capitals ("vitamins B12, D and E").
9. Singular vs. plural – Match quantity implied by the record.
10. Redundancy – Avoid repeating equivalent info ("boneless" already
    implies "without bone").
11. Brand & flavour pruning – Omit brand names; collapse long flavour
    enumerations to "assorted fruit flavours", etc.
12. Proofreading – Collapse multiple spaces, trim trailing spaces, read
    aloud to ensure the phrase is not a word salad.
13. Use 'homemade' (or similar) ONLY when the source explicitly indicates it. Otherwise leave it out.
14. Keep the original primary ingredient/species – never replace with an unrelated U.S. substitute. But you can replace the food item name if it's analagous to a US food item.
15. Remove vague packaging words ('pack', 'package', 'jar') unless nutritionally relevant (e.g. 'oil-packed tuna').
16. Assume items are commercial unless 'homemade' is explicit; use 'commercial' only when contrasting with 'homemade'.
17. If the food item is not a food item, set the unlikely_food_item flag to True.
"""

from enum import Enum
from typing import ClassVar, Optional

from pydantic import Field, validator

from ...Utils.base_food_item import FoodItem


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
    Dietary_Supplements = "Dietary Supplements"


class SR_LegacyFoodItem(FoodItem):
    description: str = Field(
        ...,
        description="The name of the food item in English. Remove brand names and other non-descriptive words. e.g. 'Coca Cola' should be 'Cola'.",
    )
    food_category: USDA_diet_categories = Field(
        ..., description="The category of the food item in English."
    )
    common_name: Optional[str] = Field(
        None,
        description="Common names associated with a food. Could be some commonly used aggragation or just the common name of the food item in English.",
    )
    unique_food_column: ClassVar[str] = "description"

    @validator("description", "common_name")
    def check_length(cls, v):
        return FoodItem.check_lengths_base(v)

    def __str__(self):
        cls_str = f"Description: {self.description}."
        cls_str += f"\nCategory: {self.food_category.value}."
        if self.common_name is not None:
            cls_str += f"\nCommon Name: {self.common_name}."
        return cls_str

    def simplify(self):
        self.food_category = None
        self.common_name = None
