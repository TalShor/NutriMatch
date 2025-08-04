import argparse
import os

import pandas as pd
from src.alignments.alignments import top_n_matches
from src.base_classes.base_data_digestion import BaseDataDigestion
from src.gpt_tools.comparison import select_closest_dataframe


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fcdb1", type=str, required=True)
    parser.add_argument("--fcdb2", type=str, required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_threads", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dataset1 = BaseDataDigestion(fcdb_name=args.fcdb1)
    dataset2 = BaseDataDigestion(fcdb_name=args.fcdb2)

    comparison_dir = f"data/comparison/top_5_matches/{args.fcdb1}_{args.fcdb2}"
    os.makedirs(comparison_dir, exist_ok=True)

    comparison_path_1in2 = (
        f"{comparison_dir}/top_n_matches_{args.fcdb1}_{args.fcdb2}.parquet"
    )
    comparison_path_2in1 = (
        f"{comparison_dir}/top_n_matches_{args.fcdb2}_{args.fcdb1}.parquet"
    )

    if not os.path.exists(comparison_path_1in2) or not os.path.exists(
        comparison_path_2in1
    ):
        (
            single_fcdb1_to_multiple_fcdb2_matches,
            single_fcdb2_to_multiple_fcdb1_matches,
        ) = top_n_matches(dataset1, dataset2, n=args.n)
        single_fcdb1_to_multiple_fcdb2_matches.to_parquet(comparison_path_1in2)
        single_fcdb2_to_multiple_fcdb1_matches.to_parquet(comparison_path_2in1)
    else:
        single_fcdb1_to_multiple_fcdb2_matches = pd.read_parquet(comparison_path_1in2)
        single_fcdb2_to_multiple_fcdb1_matches = pd.read_parquet(comparison_path_2in1)

    # get the SR Legacy columns for both of them before the comparison.

    single_fcdb1_to_multiple_fcdb2_matches = (
        single_fcdb1_to_multiple_fcdb2_matches.assign(
            str_const=lambda df: df[args.fcdb1].astype(str)
        )
        .sort_values(by="str_const")
        .head(2000)
        .drop(columns=["str_const"])
    ).sort_values(by="embedding_similarity", ascending=False)

    single_fcdb2_to_multiple_fcdb1_matches = (
        single_fcdb2_to_multiple_fcdb1_matches.assign(
            str_const=lambda df: df[args.fcdb2].astype(str)
        )
        .sort_values(by="str_const")
        .head(2000)
        .drop(columns=["str_const"])
    ).sort_values(by="embedding_similarity", ascending=False)

    single_fcdb2_to_multiple_fcdb1_matches_ranks = select_closest_dataframe(
        single_fcdb2_to_multiple_fcdb1_matches,
        col_item_a=args.fcdb2,
        col_item_b=args.fcdb1,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
    )

    # compare the top 5 matches with GPT
    single_fcdb1_to_multiple_fcdb2_matches_ranks = select_closest_dataframe(
        single_fcdb1_to_multiple_fcdb2_matches,
        col_item_a=args.fcdb1,
        col_item_b=args.fcdb2,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
    )

    single_fcdb1_to_multiple_fcdb2_matches_ranks.to_parquet(
        f"{comparison_dir}/top_n_matches_{args.fcdb1}_{args.fcdb2}_with_decision.parquet"
    )

    single_fcdb2_to_multiple_fcdb1_matches_ranks.to_parquet(
        f"{comparison_dir}/top_n_matches_{args.fcdb2}_{args.fcdb1}_with_decision.parquet"
    )

    # print(top_n_matches_1in2_boolean)


"""
pd.set_option('display.max_columns', 7)
pd.set_option('display.max_colwidth', 200)
top_n_matches_1in2_boolean.groupby('nutritionally_equivalent').sample(1).T

top_n_matches_1in2_boolean\
    [top_n_matches_1in2_boolean['SR_Legacy'].apply(lambda x: x['description'])==top_n_matches_1in2_boolean['Zameret'].apply(lambda x: x['description']) ]\
    ['nutritionally_equivalent'].mean()

top_n_matches_1in2_boolean.groupby('nutritionally_equivalent')['embedding_similarity'].describe()

"""
