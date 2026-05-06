import pandas as pd
import os

def load_multiclassdata():
    directory = "tusimple_multiclass"
    train_path = os.path.join(directory, "train")
    valid_path = os.path.join(directory, "valid")
    test_path = os.path.join(directory, "test")

    train_df = pd.read_csv(os.path.join(train_path, "metadata.csv"))
    train_df.reset_index(drop=True)
    train_df['directory'] = str(train_path)
    print(train_df.head())

    valid_df = pd.read_csv(os.path.join(valid_path, "metadata.csv"))
    valid_df.reset_index(drop=True)
    valid_df['directory'] = str(valid_path)

    test_df = pd.read_csv(os.path.join(test_path, "metadata.csv"))
    test_df.reset_index(drop=True)
    test_df['directory'] = str(test_path)
    return train_df, valid_df, test_df
