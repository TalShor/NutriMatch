import os

import pandas as pd

from ..FCDBs.SR_legacy.dataset_structure import SR_LegacyFoodItem
from ..FCDBs.Zameret.dataset_structure import ZameretFoodItem
from ..gpt_tools.embeddings import get_batch_embedding
from ..gpt_tools.translation import get_translation


class BaseDataDigestion:
    """Base class for FCDB data digestion pipelines.

    Subclasses should implement `download_raw_data` and `standardise_data`.
    The optional *data* argument is kept for backward-compatibility but is
    currently unused by the base implementation.
    """

    def __init__(self, fcdb_name: str, batch_size: int = 50, num_threads: int = 16):

        self.batch_size = batch_size
        self.num_threads = num_threads

        if fcdb_name == "SR_legacy":
            self.food_item_class: type[SR_LegacyFoodItem] = SR_LegacyFoodItem
        elif fcdb_name == "Zameret":
            self.food_item_class: type[ZameretFoodItem] = ZameretFoodItem
        else:
            raise ValueError(f"Invalid FCDB name: {fcdb_name}")

        # Prepare output directory structure (one folder per FCDB).
        self.data_base_dir = f"data/FCDBs/{fcdb_name}"
        self.code_base_dir = f"src/FCDBs/{fcdb_name}"
        self.raw_data_dir = os.path.join(self.data_base_dir, "RawData")
        os.makedirs(self.raw_data_dir, exist_ok=True)

        # Ensure GPTData directory exists early so downstream code can rely on it.
        self.gpt_data_dir = os.path.join(self.data_base_dir, "GPTData")
        self.translate_temp_dir = os.path.join(self.gpt_data_dir, "translate_temp")
        os.makedirs(self.translate_temp_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # 1. Download → cache raw data
        # ------------------------------------------------------------------
        if not os.path.exists(os.path.join(self.raw_data_dir, "raw_data.parquet")):
            self.raw_data: pd.DataFrame = self.download_raw_data()
            self.raw_data.to_parquet(
                os.path.join(self.raw_data_dir, "raw_data.parquet")
            )
        else:
            self.raw_data: pd.DataFrame = pd.read_parquet(
                os.path.join(self.raw_data_dir, "raw_data.parquet")
            )

        # ------------------------------------------------------------------
        # 2. Standardise → cache standardised data
        # ------------------------------------------------------------------
        if not os.path.exists(
            os.path.join(self.raw_data_dir, "standardised_data.parquet")
        ):
            self.standardised_data: pd.DataFrame = self.standardise_data()
            self.standardised_data.to_parquet(
                os.path.join(self.raw_data_dir, "standardised_data.parquet")
            )
        else:
            self.standardised_data: pd.DataFrame = pd.read_parquet(
                os.path.join(self.raw_data_dir, "standardised_data.parquet")
            )

        # ------------------------------------------------------------------
        # 3. Few-shot examples – must be present for GPT translation
        # ------------------------------------------------------------------
        self.few_shot_path = os.path.join(self.code_base_dir, "few_shots.yaml")
        if not os.path.exists(self.few_shot_path):
            raise FileNotFoundError(f"Few shot file not found at {self.few_shot_path}")

        # ------------------------------------------------------------------
        # 4. GPT translation (cached)
        # ------------------------------------------------------------------

        # TODO: check if it's SR_Legacy - do something else.

        if not os.path.exists(
            os.path.join(self.gpt_data_dir, "translated_data.parquet")
        ):
            self.translated_data: pd.DataFrame = self.translate_data()
            self.translated_data.to_parquet(
                os.path.join(self.gpt_data_dir, "translated_data.parquet")
            )
        else:
            self.translated_data: pd.DataFrame = pd.read_parquet(
                os.path.join(self.gpt_data_dir, "translated_data.parquet")
            )

        # ------------------------------------------------------------------
        # 5. Embeddings
        # ------------------------------------------------------------------
        if not os.path.exists(os.path.join(self.gpt_data_dir, "embeddings.parquet")):
            self.embeddings: pd.DataFrame = self.get_embeddings()
            self.embeddings.to_parquet(
                os.path.join(self.gpt_data_dir, "embeddings.parquet")
            )
        else:
            self.embeddings: pd.DataFrame = pd.read_parquet(
                os.path.join(self.gpt_data_dir, "embeddings.parquet")
            )

    def download_raw_data(self) -> pd.DataFrame:
        """Retrieve the raw dataset and return it as a DataFrame."""
        raise NotImplementedError

    def standardise_data(self) -> pd.DataFrame:
        """Transform raw data into a standardised form and return as DataFrame."""
        raise NotImplementedError

    def translate_data(self) -> pd.DataFrame:
        """Translate the standardised data using GPT."""

        # Keep only relevant columns / rows for translation.
        relevant_df = self.food_item_class.get_fields_only_df(self.standardised_data)

        # relevant_columns = relevant_columns.head(1000)

        translated_df = get_translation(
            self.few_shot_path,
            relevant_df,
            self.batch_size,
            num_threads=self.num_threads,
            temp_dir=self.translate_temp_dir,
        )

        # A long way to keep the original values while not failing the translation.
        copy_standardised_data = self.standardised_data.copy()
        full_df = pd.concat(
            [
                copy_standardised_data,
                self.food_item_class.get_fields_only_df(
                    copy_standardised_data, keep_na=True
                ).rename(
                    columns={
                        col: col + "_translated"
                        for col in copy_standardised_data.columns
                    }
                ),
            ],
            axis=1,
        )
        full_df = full_df.merge(
            translated_df.rename(
                columns={
                    col: col + "_translated"
                    for col in translated_df.columns
                    if not col.endswith("_SR_LegacyFoodItem")
                }
            ),
            how="outer",
        )

        # remove _translated columns
        full_df = full_df.drop(
            columns=[col for col in full_df.columns if col.endswith("_translated")]
        )

        return full_df

    def get_embeddings(self) -> pd.DataFrame:
        """Get the embeddings for the translated data."""
        only_sr_columns = (
            self.translated_data.filter(regex="_SR_LegacyFoodItem$")
            .rename(
                columns={
                    col: col.replace("_SR_LegacyFoodItem", "")
                    for col in self.translated_data.columns
                }
            )
            .dropna()
        )

        only_sr_columns = SR_LegacyFoodItem.df2fooditems(only_sr_columns)

        only_sr_columns = get_batch_embedding(
            only_sr_columns, num_threads=self.num_threads
        )
        self.translated_data["embedding"] = only_sr_columns
        return self.translated_data
