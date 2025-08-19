# Foundation Foods FCDB Implementation

This module implements the USDA Foundation Foods database for the NutriMatch system.

## Overview

Foundation Foods is a comprehensive database from the USDA that provides nutrient data for a wide variety of foods. This implementation enables the NutriMatch system to work with Foundation Foods data for food comparison and translation tasks.

## Dataset Information

- **Source**: USDA FoodData Central
- **Dataset**: Foundation Foods
- **URL**: https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_foundation_food_json_2025-04-24.zip
- **Foods**: 340 foundation food items
- **Nutrients**: 227 different nutrients per food item
- **Categories**: 19 food categories

## Food Categories

The Foundation Foods database includes the following categories:

- Baked Products
- Beef Products
- Beverages
- Cereal Grains and Pasta
- Dairy and Egg Products
- Fats and Oils
- Finfish and Shellfish Products
- Fruits and Fruit Juices
- Lamb, Veal, and Game Products
- Legumes and Legume Products
- Nut and Seed Products
- Pork Products
- Poultry Products
- Restaurant Foods
- Sausages and Luncheon Meats
- Soups, Sauces, and Gravies
- Spices and Herbs
- Sweets
- Vegetables and Vegetable Products

## Usage

### Basic Usage

```python
from src.base_classes.base_data_digestion import BaseDataDigestion

# Initialize Foundation Foods dataset
foundation_foods = BaseDataDigestion(fcdb_name="FoundationFoods")

# Access raw data
print(foundation_foods.raw_data.shape)

# Access standardized nutrient data
print(foundation_foods.standardised_data.shape)
```

### Command Line Usage

```bash
# Process Foundation Foods dataset
python digest_FCDB.py FoundationFoods

# Compare Foundation Foods with other databases
python comprae_FCDB.py --fcdb1 FoundationFoods --fcdb2 SR_Legacy
python comprae_FCDB.py --fcdb1 FoundationFoods --fcdb2 Zameret
```

### Using FoundationFoodsFoodItem

```python
from src.FCDBs.FoundationFoods.dataset_structure import FoundationFoodsFoodItem, USDA_foundation_categories

# Create a Foundation Foods food item
food_item = FoundationFoodsFoodItem(
    description="Beef, ground, 80% lean meat / 20% fat, raw",
    food_category=USDA_foundation_categories.Beef_Products
)

print(food_item.description)  # beef, ground, 80% lean meat 20% fat, raw
print(food_item.food_category.value)  # Beef Products
```

## Files

- `dataset_structure.py`: Defines the `FoundationFoodsFoodItem` class and food categories
- `digest_raw_data.py`: Implements data download and processing logic
- `few_shots.yaml`: Examples for GPT translation to Foundation Foods format
- `README.md`: This documentation file

## Requirements

- pandas
- requests
- pydantic
- OpenAI API key (for translation and embedding features)

## Verification

To verify the implementation is working correctly:

```bash
python verify_foundation_foods.py
```

This will test:
- Import functionality
- Class creation and validation
- Data source accessibility
- Compatibility with existing infrastructure

## Integration

Foundation Foods integrates seamlessly with the existing NutriMatch infrastructure:

- Compatible with all comparison and alignment tools
- Uses the same base classes and interfaces
- Supports the same translation and embedding workflows
- Can be compared against any other FCDB in the system

## Notes

- Foundation Foods data is automatically downloaded from USDA FoodData Central
- The data is processed and standardized to match the NutriMatch format
- Translation to SR_Legacy format is supported for cross-database comparison
- All 227 nutrients are preserved in the standardized dataset 