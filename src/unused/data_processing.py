import pandas as pd
from sklearn.model_selection import StratifiedKFold

def create_stratified_folds(csv_path, target_col="primary_label", n_splits=5, random_state=42):
    """
    Reads the CSV and adds a 'kfold' column with fold assignments using StratifiedKFold.
    Rare classes can be handled separately if desired.
    """
    df = pd.read_csv(csv_path)
    df = df.reset_index(drop=True)
    df["kfold"] = -1

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (_, val_idx) in enumerate(skf.split(X=df, y=df[target_col])):
        df.loc[val_idx, "kfold"] = fold

    return df

if __name__ == "__main__":
    CSV_PATH = "data/train.csv"
    df = create_stratified_folds(CSV_PATH, target_col="primary_label", n_splits=5, random_state=42)
    print(df.head())
    # Optionally, save the new CSV with fold information
    df.to_csv("data/train_folds.csv", index=False)