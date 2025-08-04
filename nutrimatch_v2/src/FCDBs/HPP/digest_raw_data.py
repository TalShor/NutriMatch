import pandas as pd

from src.base_classes.base_data_digestion import BaseDataDigestion


class HPPDataDigestion(BaseDataDigestion):
    def __init__(self, *args, **kwargs):
        super().__init__("HPP", *args, **kwargs)

    def download_raw_data(self) -> pd.DataFrame:
        return pd.read_parquet("data/FCDBs/HPP/RawData/FoodItems.parquet")

    def standardise_data(self) -> pd.DataFrame:
        # no nutrients in the github - but you can get it from the data access request!
        return self.raw_data[
            ["hebrew_name", "short_description", "category_hint"]
        ].reset_index(drop=True)
