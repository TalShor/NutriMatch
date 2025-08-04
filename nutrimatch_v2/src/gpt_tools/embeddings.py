import os
from typing import List

import openai
import pandas as pd
from pandarallel import pandarallel
from tenacity import retry, stop_after_attempt, wait_random_exponential

from ..base_classes.base_food_item import FoodItem

# Create a single client instance – automatically picks up the OPENAI_API_KEY env var

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@retry(
    wait=wait_random_exponential(min=1, max=20),  # exponential back-off
    stop=stop_after_attempt(3),  # give up after 3 failed attempts
    reraise=True,
)
def get_single_embedding(
    food_item: str,
    model: str = "text-embedding-3-large",
) -> List[float]:
    """Return the embedding vector for *text* using the specified OpenAI model.

    The function is resilient to transient API errors thanks to the tenacity
    retry wrapper and follows OpenAI's latest usage pattern:

        >>> vec = get_embedding("An apple a day keeps the doctor away")
        >>> len(vec)
        3072  # for text-embedding-3-large
    """

    # The API supports both a single string or a list; we pass a single str.
    response = client.embeddings.create(input=str(food_item), model=model)
    return response.data[0].embedding


def get_batch_embedding(
    food_items: List[FoodItem],
    num_threads: int = 16,
) -> List[List[float]]:
    """Compute embeddings for a list/series of FoodItem objects in parallel."""

    # Convert to pandas Series to leverage pandarallel
    str_food_items = pd.Series(food_items).apply(str)

    pandarallel.initialize(nb_workers=num_threads, progress_bar=False)
    return str_food_items.parallel_apply(get_single_embedding).tolist()
