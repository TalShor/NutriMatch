import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.base_classes.base_data_digestion import BaseDataDigestion
from src.FCDBs.SR_legacy.dataset_structure import SR_LegacyFoodItem


def matched_matrix(
    dataset1: BaseDataDigestion, dataset2: BaseDataDigestion
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """
    Create a matched matrix between two datasets.
    The matrix is a square matrix with the food items from dataset1 as the rows and the food items from dataset2 as the columns.
    The values are the cosine similarity between the embeddings of the food items.

    Args:
        dataset1 (BaseDataDigestion): The first dataset.
        dataset2 (BaseDataDigestion): The second dataset.
    Returns:
        (pd.DataFrame) A square matrix with the food items from dataset1 as the rows and the food items from dataset2 as the columns.
        The values are the cosine similarity between the embeddings of the food items.
    """

    emb_list = []
    dataset_food_items_list = []
    for dataset in [dataset1, dataset2]:
        emb = dataset.embeddings.dropna(subset=["embedding"])
        emb_list.append(np.stack(emb["embedding"]))

        # replace with SR Legacy Food Item

        emb = emb.filter(regex="_SR_LegacyFoodItem$").rename(
            columns={col: col.replace("_SR_LegacyFoodItem", "") for col in emb.columns}
        )

        dataset_food_items = SR_LegacyFoodItem.get_fields_only_df(
            emb, keep_na=True, replace_enum_to_value=True
        )
        dataset_food_items.drop(
            columns=["unlikely_food_item", "food_item_discrepancy"],
            errors="ignore",
            inplace=True,
        )
        # dataset_food_items.rename(
        #     columns={
        #         col: f"{col}_{dataset.fcdb_name}" for col in dataset_food_items.columns
        #     },
        #     inplace=True,
        # )
        dataset_food_items = dataset_food_items.apply(lambda row: row.to_dict(), axis=1)
        dataset_food_items_list.append(dataset_food_items)

    distances = cosine_similarity(
        emb_list[0],
        emb_list[1],
    )

    distances = pd.DataFrame(
        distances,
        index=dataset_food_items_list[0],
        columns=dataset_food_items_list[1],
    )
    return distances


def top_n_matches(
    dataset1: BaseDataDigestion,
    dataset2: BaseDataDigestion,
    n: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get the top n matches for each food item in dataset1 and dataset2.

    Args:
        dataset1 (BaseDataDigestion): The first dataset.
        dataset2 (BaseDataDigestion): The second dataset.
        distances (pd.DataFrame): The distances between the food items in the two datasets.
        n (int): The number of top matches to return.
    Returns:
        (pd.DataFrame) A dataframe with the top n matches for each food item in dataset1 and dataset2.
        The columns are the food items from dataset1 and dataset2, and the values are the cosine similarity between the embeddings of the food items.
    """

    distances = matched_matrix(dataset1, dataset2)
    distances = (
        distances.melt(
            ignore_index=False,
            var_name=dataset2.fcdb_name,
            value_name="embedding_similarity",
        )
        .reset_index()
        .rename(columns={"index": dataset1.fcdb_name})
        .sort_values(by="embedding_similarity", ascending=False)
    )

    def get_n_matches(
        df: pd.DataFrame,
        n: int,
        main_dataset_name: str,
    ) -> pd.DataFrame:

        return (
            df.assign(str_const=df[main_dataset_name].astype(str))
            .groupby("str_const")
            .head(n)
            .reset_index(drop=True)
            .drop(columns=["str_const"])
        )

    dataset1_top_n_matches = get_n_matches(
        distances.sort_values(by="embedding_similarity", ascending=False),
        n,
        dataset1.fcdb_name,
    )
    dataset2_top_n_matches = get_n_matches(
        distances.sort_values(by="embedding_similarity", ascending=False),
        n,
        dataset2.fcdb_name,
    )
    return dataset1_top_n_matches, dataset2_top_n_matches
