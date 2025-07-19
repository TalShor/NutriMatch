import pandas as pd
from src.gpt_tools.translation import get_batch_embedding

if __name__ == "__main__":
    batch_food_items = pd.read_parquet(
        "data/FCDBs/Zameret/RawData/standardised_data.parquet"
    )[["english_name", "hebrew_name"]].head(20)
    res = get_batch_embedding(
        batch_food_items,
        "src/FCDBs/Zameret/few_shots.yaml",
    )
    print(res)
