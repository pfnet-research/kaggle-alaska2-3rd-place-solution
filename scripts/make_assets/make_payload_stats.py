import pandas as pd
from tqdm import tqdm

from alaska2.loading import read_image

# For train
quality_df = pd.read_csv("data/working/quality_train.csv")
rows = []
for name in tqdm(quality_df["filename"]):
    cover_dct = read_image(f"data/input_nfs/Cover/{name}", "DCT")
    row = {}
    row["nonzero"] = (cover_dct != 0).sum()
    for stego in ["JMiPOD", "JUNIWARD", "UERD"]:
        stego_dct = read_image(f"data/input_nfs/{stego}/{name}", "DCT")
        row[f"{stego}_count"] = (cover_dct != stego_dct).sum()
        row[f"{stego}_count_per_nonzero"] = row[f"{stego}_count"] / row["nonzero"]
    rows.append(row)

rows = pd.DataFrame(rows)
rows.to_csv("data/working/payload_stats_train.csv", index=False)


# For test
quality_df = pd.read_csv("data/working/quality_test.csv")

rows = []
for name in tqdm(quality_df["filename"]):
    cover_dct = read_image(f"data/input_nfs/Test/{name}", "DCT")
    row = {}
    row["nonzero"] = (cover_dct != 0).sum()
    rows.append(row)

rows = pd.DataFrame(rows)
rows.to_csv("data/working/payload_stats_test.csv", index=False)
