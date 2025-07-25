import re
from enum import Enum
from typing import List, Literal, Type

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class FoodItem(BaseModel):
    unlikely_food_item: bool = Field(
        default=False,
        description="A flag to indicate that the food item is unlikely to be a food item. This is useful for filtering out non-food items from the dataset. e.g. dog milk.",
    )

    food_item_discrepancy: Literal[
        "not same food item",
        "likely not same food item",
        "likely same food item",
        "same food item",
    ] = Field(
        default="same food item",
        description="This field is used to indicate the consistency between different fields of a food item. If all fields (like description, category, etc.) refer to the same food item, the value should be 'same food item'. If there are discrepancies, choose from 'likely not same food item' or 'not same food item'. For example, if the description is 'Avocado' and the category is 'vegetables and fruits', the value should be 'same food item'. But if the description is 'Avocado' and the category is 'oil', the value should be 'not same food item'.",
    )

    def simplify(self):
        raise NotImplementedError

    @classmethod
    def df2fooditems(cls: Type["FoodItem"], df: pd.DataFrame) -> List["FoodItem"]:
        # get which attributes we added (all the fields)
        required_attrs = [attr for attr in cls.model_fields]
        required_attrs = np.setdiff1d(
            required_attrs, list(FoodItem.model_fields.keys())
        )

        print("cols")
        print(df.columns)
        print(required_attrs)

        if df.columns.intersection(required_attrs).size != len(required_attrs):
            raise ValueError(
                f"DataFrame does not contain all the required attributes for {cls}."
            )

        # doesn't work with np.nan
        df.replace({np.nan: None}, inplace=True)
        return df[required_attrs].apply(lambda row: cls(**row.to_dict()), axis=1)

    @classmethod
    def fooditems2df(cls: Type["FoodItem"], food_items):
        if isinstance(food_items, pd.Series):
            food_items = food_items.tolist()

        # Extract data using the actual class of each item, accommodating for subclass attributes
        data = [
            {field: getattr(item, field, None) for field in item.__fields__}
            for item in food_items
        ]

        return pd.DataFrame(data)

    @classmethod
    def get_fields_only_df(
        cls: Type["FoodItem"],
        df: pd.DataFrame,
        keep_na: bool = False,
        replace_enum_to_value: bool = False,
    ) -> pd.DataFrame:
        """
        This function is used to get the fields only from the dataframe.
        It is used to get the fields only from the dataframe.
        If the row is not valid - drop it.
        """
        only_fields_df = cls.df2fooditems(df)
        only_fields_df = cls.fooditems2df(only_fields_df)

        if not keep_na:
            for field in cls.model_fields.keys():
                if cls.model_fields[field].is_required():
                    only_fields_df = only_fields_df[only_fields_df[field].notna()]

            only_fields_df = only_fields_df.drop(
                columns=["unlikely_food_item", "food_item_discrepancy"], errors="ignore"
            )

        if replace_enum_to_value:
            for field in cls.model_fields.keys():
                annotation = cls.model_fields[field].annotation
                if isinstance(annotation, type) and issubclass(annotation, Enum):
                    only_fields_df[field] = only_fields_df[field].apply(
                        lambda x: x.value
                    )

        return only_fields_df

    @classmethod
    def add_fcdb_to_columns(cls: Type["FoodItem"], df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns={col: f"{col}_{cls.__name__}" for col in df.columns})

    def check_lengths_base(str_validation: str) -> str:
        if str_validation is None:
            return str_validation

        cleaned_str = re.sub(r"[\,]+", ",", str_validation)
        cleaned_str = re.sub(r"[\.]+", ".", cleaned_str)
        cleaned_str = (
            cleaned_str.replace("\n", "")
            .replace("\t", " ")
            .replace("-", " ")
            .replace("\\", " ")
            .replace("/", " ")
            .replace(",", ", ")
            .replace(".", ". ")
            .replace(":", " ")
            .strip()
            .lower()
        )
        cleaned_str = re.sub(r"\s+", " ", cleaned_str)

        # no words over 20 characters
        if any(len(word) > 20 for word in cleaned_str.split()):
            print("no words over 20 characters")
            return None

        if len(cleaned_str) < 2 or len(cleaned_str) > 200:
            print(
                f"description must be between 2 and 200 characters for the word {str_validation} -> {cleaned_str}"
            )
            return None

        return cleaned_str
