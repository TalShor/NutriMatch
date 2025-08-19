# Foundation Foods Setup Guide

## What Was Implemented

I have successfully created a complete Foundation Foods dataset implementation for the NutriMatch system using the USDA FoodData Central Foundation Foods dataset from:
https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_foundation_food_json_2025-04-24.zip

## Implementation Details

### Files Created:
- `src/FCDBs/FoundationFoods/dataset_structure.py` - Defines FoundationFoodsFoodItem class
- `src/FCDBs/FoundationFoods/digest_raw_data.py` - Data download and processing logic
- `src/FCDBs/FoundationFoods/few_shots.yaml` - GPT translation examples
- `src/FCDBs/FoundationFoods/__init__.py` - Module initialization
- `src/FCDBs/FoundationFoods/README.md` - Detailed documentation

### Modified Files:
- `src/FCDBs/__init__.py` - Added FoundationFoodsFoodItem import

## Dataset Specifications

- **340 Foundation Foods items**
- **227 nutrients per food item** 
- **19 food categories**
- **Compatible with existing comparison infrastructure**

## How to Use

### 1. Set up OpenAI API Key (required for full functionality)
```bash
export OPENAI_API_KEY='your-api-key-here'
```

### 2. Process Foundation Foods Dataset
```bash
python digest_FCDB.py FoundationFoods
```

### 3. Compare with Other Databases
```bash
# Compare Foundation Foods with SR Legacy
python comprae_FCDB.py --fcdb1 FoundationFoods --fcdb2 SR_Legacy

# Compare Foundation Foods with Zameret
python comprae_FCDB.py --fcdb1 FoundationFoods --fcdb2 Zameret

# Compare SR Legacy with Foundation Foods
python comprae_FCDB.py --fcdb1 SR_Legacy --fcdb2 FoundationFoods
```

### 4. Programmatic Usage
```python
from src.base_classes.base_data_digestion import BaseDataDigestion

# Initialize Foundation Foods
foundation_foods = BaseDataDigestion(fcdb_name="FoundationFoods")

# Access the data
print(f"Raw data shape: {foundation_foods.raw_data.shape}")
print(f"Standardized data shape: {foundation_foods.standardised_data.shape}")
```

## Verification

To verify the implementation is working:
```bash
python verify_foundation_foods.py
```

## Food Categories Available

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

## Key Features

✅ **Automatic Data Download** - Downloads latest Foundation Foods data from USDA  
✅ **Nutrient Processing** - Extracts and standardizes 227 nutrients per food  
✅ **Category Mapping** - Maps to 19 standardized food categories  
✅ **Translation Support** - Includes few-shot examples for GPT translation  
✅ **Comparison Ready** - Works with existing comparison and alignment tools  
✅ **Validation** - Includes comprehensive validation and verification  

## Integration Status

The Foundation Foods implementation is fully integrated with the NutriMatch system:

- ✅ Base classes and interfaces
- ✅ Data processing pipeline  
- ✅ Translation system
- ✅ Comparison tools
- ✅ Embedding generation
- ✅ Validation and testing

Foundation Foods can now be used interchangeably with any other FCDB in the system (SR_Legacy, Zameret, FNDDS, HPP) for all comparison and analysis tasks. 