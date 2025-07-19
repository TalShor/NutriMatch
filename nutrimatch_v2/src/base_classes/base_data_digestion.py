import os

import pandas as pd

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

        # Prepare output directory structure (one folder per FCDB).
        self.data_base_dir = f"data/FCDBs/{fcdb_name}"
        self.code_base_dir = f"src/FCDBs/{fcdb_name}"
        self.raw_data_dir = os.path.join(self.data_base_dir, "RawData")
        os.makedirs(self.raw_data_dir, exist_ok=True)

        # Ensure GPTData directory exists early so downstream code can rely on it.
        self.gpt_data_dir = os.path.join(self.data_base_dir, "GPTData")
        os.makedirs(self.gpt_data_dir, exist_ok=True)

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

    def download_raw_data(self) -> pd.DataFrame:
        """Retrieve the raw dataset and return it as a DataFrame."""
        raise NotImplementedError

    def standardise_data(self) -> pd.DataFrame:
        """Transform raw data into a standardised form and return as DataFrame."""
        raise NotImplementedError

    def translate_data(self) -> pd.DataFrame:
        """Translate the standardised data using GPT."""
        return get_translation(
            self.few_shot_path,
            self.standardised_data,
            self.batch_size,
            num_threads=self.num_threads,
        )
