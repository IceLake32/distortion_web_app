from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "web_app" / "example_data"


def make_raw_swiss_roll(n_samples: int = 240, seed: int = 7) -> pd.DataFrame:
    x, position = make_swiss_roll(n_samples=n_samples, noise=0.08, random_state=seed)
    x = StandardScaler().fit_transform(x)
    bands = pd.qcut(position, q=3, labels=["early", "middle", "late"])
    return pd.DataFrame(
        {
            "sample_id": [f"sample_{i:03d}" for i in range(n_samples)],
            "feature_1": x[:, 0],
            "feature_2": x[:, 1],
            "feature_3": x[:, 2],
            "label": bands.astype(str),
            "manifold_position": position,
        }
    )


def add_existing_embedding(raw: pd.DataFrame, n_neighbors: int = 14) -> pd.DataFrame:
    features = raw[["feature_1", "feature_2", "feature_3"]].to_numpy()
    y = Isomap(n_neighbors=n_neighbors, n_components=2).fit_transform(features)
    y = StandardScaler().fit_transform(y)
    out = raw.copy()
    out["isomap_1"] = y[:, 0]
    out["isomap_2"] = y[:, 1]
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = make_raw_swiss_roll()
    embedded = add_existing_embedding(raw)

    raw.to_csv(OUT_DIR / "raw_swiss_roll_example.csv", index=False)
    embedded.to_csv(OUT_DIR / "existing_embedding_swiss_roll_example.csv", index=False)

    print(f"Wrote {OUT_DIR / 'raw_swiss_roll_example.csv'}")
    print(f"Wrote {OUT_DIR / 'existing_embedding_swiss_roll_example.csv'}")


if __name__ == "__main__":
    main()
