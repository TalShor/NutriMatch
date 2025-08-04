import asyncio
import boto3
from botocore.exceptions import  ClientError
import instructor
from openai import OpenAI, AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_fixed
import json
import enum
from tqdm.asyncio import tqdm_asyncio
from pydantic import BaseModel, Field
from .base_food_item import FoodItem

def get_secret():
    """
    Retrieves the OpenAI API key from AWS Secrets Manager.

    Returns:
        dict: A dictionary containing the OpenAI API key.
    """
    secret_name = "open_ai_api_key_shahar"
    region_name = "eu-west-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    
    get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
  
    secret = get_secret_value_response['SecretString']
  
    return  json.loads(secret)

api_key = get_secret()['open_ai_key']
client = instructor.patch(OpenAI(api_key=api_key))
async_client = instructor.from_openai(
    AsyncOpenAI(api_key=api_key)
)
translation_model  = "gpt-4o"

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry_error_callback=lambda _: None)
def get_translation(food_description:FoodItem, output_format: FoodItem) -> str:
    # TODO: output_format should be a class or something
    food_item = client.chat.completions.create(
        model=translation_model, 
        messages=[{"role":"user", "content":f"Convert this food item to the given format:\n{str(food_description)}"}],
        response_model=output_format, # USDASRLegacyFoodItem
    )

    return food_item

def get_translation_parallel(food_description:str, output_format: FoodItem) -> str:
    # TODO: output_format should be a class or something
    food_item = client.chat.completions.create(
        model= translation_model, 
        messages=[{"role":"user", "content":f"Convert this food item to the given format:\n{food_description}"}],
        response_model=output_format, # USDASRLegacyFoodItem
    )
    food_item_dict = food_item.__dict__
    food_item_dict.pop('_raw_response')

    return food_item_dict

@retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry_error_callback=lambda _: None)
def get_embedding(food_item:FoodItem, model="text-embedding-3-large"):
   return client.embeddings.create(input = str(food_item), model=model)\
        .data[0].embedding



def get_embedding_parallel_generalised(food_item: dict, food_item_class, model="text-embedding-3-large"):
    food_item_instance = food_item_class(**food_item)
    return client.embeddings.create(input=str(food_item_instance), model=model).data[0].embedding




class VALIDATION_TYPE(enum.Enum):
    VALIDATED_MATCHES = "basic"
    VALIDATED_PORTION_UNIT_MATCHES = "portions"

    def get_system_prompt(self) -> str:
        if self == VALIDATION_TYPE.VALIDATED_MATCHES:
            return """
            As a registered dietitian, evaluate whether the following two food item names refer to the same item.
            Consider factors such as potential variations in naming conventions, common culinary terms, and any descriptive elements that might indicate similarity or difference."""
        elif self == VALIDATION_TYPE.VALIDATED_PORTION_UNIT_MATCHES:
#             return """
# As a registered dietitian, evaluate whether the following two food items use the same portion units (e.g., cup, tablespoons/teaspoons, slices, bowls, cans).
# Consider the typical units used for these items in everyday consumption. Provide a simple 'True' or 'False' response."""
            return """As a registered dietitian, evaluate whether the following two food items typically use the same portion units, considering that minor differences in names (such as species or preparation details) should not affect the evaluation unless they fundamentally change how the item is portioned.
            Provide a simple 'True' or 'False' response."""

    def get_user_prompt(self) -> str:
        if self == VALIDATION_TYPE.VALIDATED_MATCHES:
            return """Item 1: {food_item1}
            Item 2: {food_item2}

            Do these items refer to the same food item? True or False."""
        elif self == VALIDATION_TYPE.VALIDATED_PORTION_UNIT_MATCHES:
            return """Item 1: {food_item1}
            Item 2: {food_item2}

            Do these items use the same portion units? True or False."""
        
    def get_directory_name(self) -> str:
        return self.name.lower()

CONCURRENCY_LIMIT = 1000

async def same_food_item(semaphore: asyncio.Semaphore, index: int, food_item1: str, food_item2: str, validation_type: VALIDATION_TYPE) -> tuple[int, bool]:
    """
    Determine if two food items are the same or not.

    Args:
        semaphore (asyncio.Semaphore): A semaphore to limit the number of concurrent requests.
        index (int): The index of the food items.
        food_item1 (str): The first food item.
        food_item2 (str): The second food item.
        validation_type (VALIDATION_TYPE): The type of validation to perform.

    Returns:
        tuple[int, bool]: The index of the food items and a boolean indicating if they are the same.
    """
    
    system_prompt = validation_type.get_system_prompt()
    user_prompt = validation_type.get_user_prompt().format(food_item1=food_item1, food_item2=food_item2)
    async with semaphore:
        prediction = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            response_model=bool
        )
        return (index, prediction)

async def validate_food_items(food_items: list[tuple[str, str]], validation_type: VALIDATION_TYPE) -> list[tuple[int, bool]]:
    """
    Validates a list of food items by comparing each pair of food items using the same_food_item function.

    Args:
        food_items (list[tuple[str, str]]): A list of tuples, where each tuple contains two food items.
        validation_type (VALIDATION_TYPE): The type of validation to perform.

    Returns:
        list[tuple[int, bool]]: A list of tuples, where each tuple contains the index of the food item pair and
            a boolean value indicating whether the two food items are the same.

    """
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    predictions = await tqdm_asyncio.gather(
        *(same_food_item(semaphore, index, food_item1, food_item2, validation_type) for index, (food_item1, food_item2) in enumerate(food_items)),
        desc="Validating food matches"
    )
    return predictions
    