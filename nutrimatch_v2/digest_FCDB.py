import argparse

from src.base_classes.base_data_digestion import BaseDataDigestion
from src.FCDBs import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Digest FCDBs")
    parser.add_argument("--fcdb", type=str, required=True, help="The FCDB to digest")
    return parser.parse_args()


def get_data_digestion_class(fcdb_name: str) -> type[BaseDataDigestion]:
    try:
        return globals()[f"{fcdb_name}DataDigestion"]
    except KeyError:
        raise ValueError(f"Invalid FCDB: {fcdb_name}")


if __name__ == "__main__":
    args = parse_args()
    data_digestion_class = get_data_digestion_class(args.fcdb)
    data_digestion_class()
