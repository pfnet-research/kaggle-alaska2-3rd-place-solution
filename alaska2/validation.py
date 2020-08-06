import pandas as pd
from sklearn.model_selection import StratifiedKFold


def get_folds_as_filename(n_folds):
    quality_df = pd.read_csv("data/working/quality_train.csv")
    kf = StratifiedKFold(n_folds, shuffle=True, random_state=0)
    filenames = quality_df["filename"].to_numpy()
    return [
        (filenames[train], filenames[valid]) for train, valid in kf.split(quality_df["quality"], quality_df["quality"])
    ]
