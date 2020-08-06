from pathlib import Path

import pandas as pd
from tqdm import tqdm

from alaska2.loading import read_quality_factor


def get_quality_df(target):
    df = []
    for path in tqdm(sorted(Path(f"data/input/{target}").glob("*.jpg"))):
        df.append((path.name, read_quality_factor(path)))
    return pd.DataFrame(df, columns=["filename", "quality"])


Path("data/working/").mkdir(parents=True, exist_ok=True)

get_quality_df("Cover").to_csv("data/working/quality_train.csv", index=False)
get_quality_df("Test").to_csv("data/working/quality_test.csv", index=False)
