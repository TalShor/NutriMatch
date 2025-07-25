import os
from typing import Callable

import pandas as pd

from ..FCDBs.SR_legacy.dataset_structure import SR_LegacyFoodItem
from ..FCDBs.Zameret.dataset_structure import ZameretFoodItem
from ..gpt_tools.embeddings import get_batch_embedding
from ..gpt_tools.translation import get_translation

SR_Legacy_name = "SR_Legacy"


class BaseDataDigestion:
    """Base class for FCDB data digestion pipelines.

    Subclasses should implement `download_raw_data` and `standardise_data`.
    The optional *data* argument is kept for backward-compatibility but is
    currently unused by the base implementation.
    """

    def __save_data(self, path: str, create_func: Callable):
        if not os.path.exists(path):
            data = create_func()
            data.to_parquet(path)
            return data
        return pd.read_parquet(path)

    def __init__(self, fcdb_name: str, batch_size: int = 50, num_threads: int = 16):

        self.batch_size = batch_size
        self.num_threads = num_threads
        self.fcdb_name = fcdb_name

        if fcdb_name == SR_Legacy_name:
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
        self.raw_data = self.__save_data(
            os.path.join(self.raw_data_dir, "raw_data.parquet"), self.download_raw_data
        )

        # ------------------------------------------------------------------
        # 2. Standardise → cache standardised data
        # ------------------------------------------------------------------
        self.standardised_data = self.__save_data(
            os.path.join(self.raw_data_dir, "standardised_data.parquet"),
            self.standardise_data,
        )

        # ------------------------------------------------------------------
        # 3. Few-shot examples – must be present for GPT translation
        # ------------------------------------------------------------------
        self.few_shot_path = os.path.join(self.code_base_dir, "few_shots.yaml")
        if (self.fcdb_name != SR_Legacy_name) and not os.path.exists(
            self.few_shot_path
        ):
            raise FileNotFoundError(f"Few shot file not found at {self.few_shot_path}")

        # ------------------------------------------------------------------
        # 4. GPT translation
        # ------------------------------------------------------------------

        # If it's SR_Legacy, we need don't need to run translation - just copy the columns with a suffix.
        def sr_legacy_translation():
            df = self.standardised_data.copy().reset_index()
            df = pd.concat(
                [
                    df,
                    SR_LegacyFoodItem.add_fcdb_to_columns(df),
                ],
                axis=1,
            )
            return df

        self.translation = self.__save_data(
            os.path.join(self.gpt_data_dir, "translation.parquet"),
            (
                sr_legacy_translation
                if self.fcdb_name == SR_Legacy_name
                else self.translate_data
            ),
        )

        # ------------------------------------------------------------------
        # 5. Embeddings
        # ------------------------------------------------------------------
        self.embeddings = self.__save_data(
            os.path.join(self.gpt_data_dir, "embeddings.parquet"), self.get_embeddings
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

        only_sr_columns = self.translation.filter(regex="_SR_LegacyFoodItem$").rename(
            columns={
                col: col.replace("_SR_LegacyFoodItem", "")
                for col in self.translation.columns
            }
        )

        print(only_sr_columns.head())
        only_sr_columns = SR_LegacyFoodItem.df2fooditems(only_sr_columns)

        only_sr_columns = get_batch_embedding(
            only_sr_columns, num_threads=self.num_threads
        )
        self.translation["embedding"] = only_sr_columns
        return self.translation
