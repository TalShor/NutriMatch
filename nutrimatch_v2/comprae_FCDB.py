import argparse
import os

import pandas as pd
from src.alignments.alignments import top_n_matches
from src.base_classes.base_data_digestion import BaseDataDigestion
from src.gpt_tools.are_equal import are_equal_dataframe
from src.syntax_matching.exact_matcher import ExactFoodMatcher


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_fcdb", type=str, required=True)
    parser.add_argument("--candidate_fcdb", type=str, required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_threads", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main_dataset = BaseDataDigestion(fcdb_name=args.main_fcdb)
    candidates_dataset = BaseDataDigestion(fcdb_name=args.candidate_fcdb)

    comparison_dir = "data/comparison/top_5_matches"
    os.makedirs(comparison_dir, exist_ok=True)
    comparison_prefix = f"{comparison_dir}/{args.main_fcdb}_{args.candidate_fcdb}"

    # get the top n matches
    top_n_matches_path = comparison_prefix + "_top_n_matches.parquet"
    if not os.path.exists(top_n_matches_path):
        main_to_multiple_candidates, _ = top_n_matches(
            main_dataset, candidates_dataset, n=args.n
        )

        main_to_multiple_candidates["main_description"] = main_to_multiple_candidates[
            args.main_fcdb
        ].apply(lambda x: x["description"])
        main_to_multiple_candidates["candidate_description"] = (
            main_to_multiple_candidates[args.candidate_fcdb].apply(
                lambda x: x["description"]
            )
        )
        matcher = ExactFoodMatcher()
        main_to_multiple_candidates["exact_match"] = main_to_multiple_candidates.apply(
            lambda row: matcher.find_exact_matches(
                row["main_description"], [row["candidate_description"]]
            ),
            axis=1,
        ).str[0]

        main_to_multiple_candidates.to_parquet(top_n_matches_path)
    else:
        main_to_multiple_candidates = pd.read_parquet(top_n_matches_path)

    # split main that have an exact match and those who don't
    has_exact_match = main_to_multiple_candidates.groupby("main_description")[
        "exact_match"
    ].max()

    main_to_multiple_candidates_with_exact_match = main_to_multiple_candidates[
        main_to_multiple_candidates["main_description"].isin(
            has_exact_match[has_exact_match].index
        )
    ]

    main_to_multiple_candidates_without_exact_match = main_to_multiple_candidates[
        ~main_to_multiple_candidates["main_description"].isin(
            has_exact_match[has_exact_match].index
        )
    ]

    # compare the top 5 matches with GPT
    main_to_multiple_candidates_without_exact_match = (
        main_to_multiple_candidates_without_exact_match.sort_values(
            by="main_description"
        )
        .head(3000)
        .tail(1000)
    )

    main_to_multiple_candidates_without_exact_match_and_similarity = (
        are_equal_dataframe(
            main_to_multiple_candidates_without_exact_match,
            col_reference=args.main_fcdb,
            col_candidates=args.candidate_fcdb,
            batch_size=args.batch_size,
            num_threads=args.num_threads,
        )
    )

    results = pd.concat(
        [
            main_to_multiple_candidates_with_exact_match,
            main_to_multiple_candidates_without_exact_match_and_similarity,
        ]
    ).drop(columns=["main_description", "candidate_description"])
    results.to_parquet(f"{comparison_prefix}_top_n_matches_with_decision.parquet")
