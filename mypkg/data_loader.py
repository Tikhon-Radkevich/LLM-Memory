import pandas as pd


dataset_path = "/kaggle/input/datasets/tikhonradk/retrieval-question-answering"


def load_data():
    train = pd.read_json(f"{dataset_path}/train.json", lines=True)
    test = pd.read_json(f"{dataset_path}/test.json", lines=True)
    return train, test

def preview_data(train_df: pd.DataFrame):
    pid, question, options, correct_indices, wrong_indices = train_df.iloc[40]

    print('QUESTION', question, '\n')
    print('TEXT SENTENCES')
    for i, cand in enumerate(options):
        print(['[ ]', '[v]'][i in correct_indices], cand)
