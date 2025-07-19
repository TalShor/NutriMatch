import os

import pandas as pd


class BaseDataDigestion:
    """Base class for FCDB data digestion pipelines.

    Subclasses should implement `download_raw_data` and `standardise_data`.
    The optional *data* argument is kept for backward-compatibility but is
    currently unused by the base implementation.
    """

    def __init__(self, fcdb_name: str, batch_size: int = 50):
        # Subclasses may still pass an unused *data* argument.

        # Download and standardise data through the subclass implementations.
        self.raw_data: pd.DataFrame = self.download_raw_data()
        self.standardised_data: pd.DataFrame = self.standardise_data()

        # Prepare output directory structure.
        self.base_dir = f"nutrimatch_v2/data/FCDBs/{fcdb_name}/RawData"
        self.base_dirraw_data_dir = os.path.join(self.base_dir, "RawData")
        os.makedirs(self.base_dir, exist_ok=True)

        # check if few_shot.yaml exists
        self.few_shot_path = os.path.join(self.base_dir, "few_shot.yaml")
        if not os.path.exists(self.few_shot_path):
            raise FileNotFoundError(f"Few shot file not found at {self.few_shot_path}")

        # Persist the data for later analysis.
        self.raw_data.to_parquet(os.path.join(self.raw_data_dir, "raw_data.parquet"))
        self.standardised_data.to_parquet(
            os.path.join(self.raw_data_dir, "standardised_data.parquet")
        )

        self.batch_size = batch_size

    def download_raw_data(self) -> pd.DataFrame:
        """Retrieve the raw dataset and return it as a DataFrame."""
        raise NotImplementedError

    def standardise_data(self) -> pd.DataFrame:
        """Transform raw data into a standardised form and return as DataFrame."""
        raise NotImplementedError
