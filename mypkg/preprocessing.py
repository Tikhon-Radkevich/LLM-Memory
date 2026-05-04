from itertools import chain

import pandas as pd


def build_vocab(dfs: list[pd.DataFrame]) -> tuple[dict[str, int], dict[str, int]]:
    all_options = list(set(chain.from_iterable(df["options"].explode() for df in dfs)))
    all_questions = list(set(chain.from_iterable(df["question"].unique() for df in dfs)))

    options_to_idx = {opt: i for i, opt in enumerate(all_options)}
    questions_to_idx = {q: i for i, q in enumerate(all_questions)}
    return options_to_idx, questions_to_idx


def process_split(
    df: pd.DataFrame,
    options_to_idx: dict[str, int],
    questions_to_idx: dict[str, int],
) -> pd.DataFrame:
    df = df.copy()
    df["question_id"] = df["question"].map(questions_to_idx)
    df["options_ids"] = df["options"].map(lambda opts: [options_to_idx[o] for o in opts])
    df["wrong_options_ids"] = df.apply(
        lambda row: [row["options_ids"][i] for i in row["wrong_indices"]], axis=1
    )
    df["correct_option_id"] = df.apply(
        lambda row: row["options_ids"][row["correct_indices"][0]], axis=1
    )
    return df


def preprocess(train: pd.DataFrame, test: pd.DataFrame):
    options_to_idx, questions_to_idx = build_vocab([train, test])
    train = process_split(train, options_to_idx, questions_to_idx)
    test = process_split(test, options_to_idx, questions_to_idx)
    return train, test, options_to_idx, questions_to_idx
