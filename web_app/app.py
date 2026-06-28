from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_blobs, make_s_curve, make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    umap = None
    UMAP_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
for package_root in (ROOT, ROOT / "distortions"):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from distortions.geometry import Geometry, riemann_metric  # noqa: E402


st.set_page_config(
    page_title="Distortions Demo",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


DATASET_HELP = {
    "Swiss roll": "A folded 2D manifold in 3D where unfolding should preserve local geometry.",
    "S-curve": "A smoother 3D manifold with visible bends and local stretching.",
    "Two clusters": "Two nearby high-dimensional groups where embeddings can exaggerate gaps.",
}

COLOR_HELP = {
    "distortion_ratio": "Color shows anisotropic local stretching: brighter points have more elongated local metric ellipses.",
    "local_area": "Color shows the overall local metric scale: brighter points have larger local expansion/compression magnitude.",
    "reference": "Color shows the selected column. Numeric columns use a continuous scale; categorical columns use distinct colors.",
}

COLOR_LABELS = {
    "distortion_ratio": "distortion ratio",
    "local_area": "local area",
    "reference": "selected column",
}

EXAMPLE_DIR = Path(__file__).resolve().parent / "example_data"
RAW_DATA_EXAMPLE_PATH = EXAMPLE_DIR / "raw_swiss_roll_example.csv"
EMBEDDING_DATA_EXAMPLE_PATH = EXAMPLE_DIR / "existing_embedding_swiss_roll_example.csv"


def render_upload_format_guide() -> None:
    st.subheader("Upload data format")
    st.write(
        "Upload a table where each row is one sample. Numeric columns can be used as the "
        "original high-dimensional features. Column names do not need to match the examples; "
        "the app will suggest sensible defaults and you can adjust them if needed. You can "
        "either let the app compute an embedding, or provide existing embedding coordinate columns."
    )

    st.markdown("**Supported files:** CSV, TSV/TXT, XLSX, and XLS.")

    compute_tab, embedding_tab = st.tabs(["App computes embedding", "Use uploaded embedding"])
    with compute_tab:
        st.write("Use this format when the app should compute PCA, Isomap, t-SNE, or UMAP.")
        st.code(
            """feature_1,feature_2,feature_3,label
0.1,1.2,0.4,A
0.3,1.0,0.5,A
2.4,0.2,1.1,B""",
            language="csv",
        )
        st.markdown(
            """
            The feature column names can be anything. Numeric columns are treated as candidate
            features. Optional label/reference columns can be categorical or numeric; they are
            used only for labels, tooltips, or reference coloring.
            """
        )

    with embedding_tab:
        st.write("Use this format when you already have UMAP, t-SNE, PCA, or other 2D/3D coordinates.")
        st.code(
            """feature_1,feature_2,feature_3,umap_1,umap_2,label
0.1,1.2,0.4,-2.1,0.5,A
0.3,1.0,0.5,-1.9,0.7,A
2.4,0.2,1.1,1.4,-0.3,B""",
            language="csv",
        )
        st.markdown(
            """
            The coordinate column names can be anything, though names like `embedding_0`,
            `embedding_1`, and `embedding_2` will be detected automatically. These embedding
            columns are excluded from the original feature matrix.
            """
        )


def results_csv(metrics: pd.DataFrame, features: pd.DataFrame | None = None) -> bytes:
    out = metrics.drop(columns=["reference_is_numeric", "has_reference"], errors="ignore").reset_index(names="sample_index")
    if features is not None:
        feature_table = features.reset_index(drop=True).copy()
        feature_table = feature_table.loc[:, [col for col in feature_table.columns if col not in out.columns]]
        out = pd.concat([feature_table, out], axis=1)
    preferred = [
        "sample_index",
        "feature_1",
        "feature_2",
        "feature_3",
        "embedding_0",
        "embedding_1",
        "embedding_2",
        "label",
        "reference_value",
        "manifold_position",
        "distortion_ratio",
        "local_area",
        "s0",
        "s1",
        "x0",
        "x1",
        "y0",
        "y1",
    ]
    columns = [col for col in preferred if col in out.columns]
    columns += [col for col in out.columns if col not in columns]
    return out.loc[:, columns].to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner=False)
def example_csv_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


@st.cache_data(show_spinner=False)
def example_preview(path: str) -> pd.DataFrame:
    return pd.read_csv(path).head(6)


RELEASE_BASE_URL = "https://github.com/IceLake32/distortion_web_app/releases/latest/download"
DISTORTIONS_PAPER_URL = "https://academic.oup.com/bib/article/27/2/bbag136/8559622"
RMETRIC_PAPER_URL = "https://arxiv.org/abs/1305.7255"
DISTORTIONS_PACKAGE_URL = "https://github.com/krisrs1128/distortions"
DISTORTIONS_DOCS_URL = "https://krisrs1128.github.io/distortions/site/"


@st.cache_data(show_spinner=False)
def read_uploaded_table(file_bytes: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    buffer = io.BytesIO(file_bytes)
    if suffix == ".csv":
        return pd.read_csv(buffer)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(buffer, sep="\t")
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(buffer)
    raise ValueError(f"Unsupported file format: {suffix}")


def _standardize_embedding(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return StandardScaler().fit_transform(y)


def make_dataset(name: str, n_samples: int, noise: float, seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    if name == "Swiss roll":
        x, color = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=seed)
        meta = pd.DataFrame({
            "label": "roll",
            "hover_text": [f"sample=roll<br>value={value:.2f}" for value in color],
            "manifold_position": color,
            "reference_value": color,
            "reference_is_numeric": True,
            "has_reference": True,
            "reference_name": "manifold_position",
        })
        return StandardScaler().fit_transform(x), meta

    if name == "S-curve":
        x, color = make_s_curve(n_samples=n_samples, noise=noise, random_state=seed)
        meta = pd.DataFrame({
            "label": "curve",
            "hover_text": [f"sample=curve<br>value={value:.2f}" for value in color],
            "manifold_position": color,
            "reference_value": color,
            "reference_is_numeric": True,
            "has_reference": True,
            "reference_name": "manifold_position",
        })
        return StandardScaler().fit_transform(x), meta

    centers = np.array([[-1.2, 0, 0, 0, 0, 0], [1.2, 0.35, 0, 0, 0, 0]])
    x, labels = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=[0.55 + noise, 0.65 + noise],
        n_features=6,
        random_state=seed,
    )
    x[:, 2:] += 0.35 * rng.normal(size=x[:, 2:].shape)
    cluster_labels = np.where(labels == 0, "cluster A", "cluster B")
    meta = pd.DataFrame({
        "label": cluster_labels,
        "hover_text": [f"value={value}" for value in cluster_labels],
        "manifold_position": labels,
        "reference_value": cluster_labels,
        "reference_is_numeric": False,
        "has_reference": True,
        "reference_name": "label",
    })
    return StandardScaler().fit_transform(x), meta


def prepare_uploaded_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str | None,
    embedding_cols: list[str] | None,
    max_rows: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray | None, pd.DataFrame, pd.Index]:
    features = df.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")
    pieces = [features]
    embedding = None
    if embedding_cols is not None:
        embedding = df.loc[:, embedding_cols].apply(pd.to_numeric, errors="coerce")
        pieces.append(embedding)

    valid = pd.concat(pieces, axis=1).replace([np.inf, -np.inf], np.nan).dropna()

    if len(valid) > max_rows:
        valid = valid.sample(n=max_rows, random_state=seed).sort_index()

    feature_export = valid.loc[:, feature_cols].reset_index(drop=True)
    x = StandardScaler().fit_transform(feature_export.to_numpy(dtype=float))
    y = None
    if embedding_cols is not None:
        y = _standardize_embedding(valid.loc[:, embedding_cols].to_numpy(dtype=float))

    if label_col is not None:
        raw_reference = df.loc[valid.index, label_col].reset_index(drop=True)
        numeric_reference = pd.to_numeric(raw_reference, errors="coerce")
        reference_is_numeric = numeric_reference.notna().all()
        if reference_is_numeric:
            reference_value = numeric_reference.astype(float)
            manifold_position = reference_value
            if "sample_id" in df.columns:
                labels = df.loc[valid.index, "sample_id"].astype(str).fillna("sample").reset_index(drop=True)
            elif "id" in df.columns:
                labels = df.loc[valid.index, "id"].astype(str).fillna("sample").reset_index(drop=True)
            else:
                labels = pd.Series([f"sample {ix}" for ix in range(len(valid))])
            hover_text = [
                f"sample={label}<br>value={value:.3g}"
                for label, value in zip(labels, reference_value)
            ]
        else:
            labels = raw_reference.astype(str).fillna("missing")
            reference_value = labels
            manifold_position = pd.factorize(labels)[0]
            hover_text = [f"value={label}" for label in labels]
    else:
        labels = pd.Series(["uploaded"] * len(valid))
        manifold_position = np.arange(len(valid))
        reference_value = labels
        reference_is_numeric = False
        hover_text = [f"sample={label}" for label in labels]
    has_reference = label_col is not None

    meta = pd.DataFrame({
        "label": labels,
        "hover_text": hover_text,
        "manifold_position": manifold_position,
        "reference_value": reference_value,
        "reference_is_numeric": reference_is_numeric,
        "has_reference": has_reference,
        "reference_name": label_col if label_col is not None else "None",
    })
    return x, meta, y, feature_export, valid.index


def apply_reference_column(meta: pd.DataFrame, raw_df: pd.DataFrame, valid_index: pd.Index, reference_col: str | None) -> pd.DataFrame:
    out = meta.copy()
    if reference_col is None:
        out["has_reference"] = False
        out["reference_name"] = "None"
        return out

    raw_reference = raw_df.loc[valid_index, reference_col].reset_index(drop=True)
    numeric_reference = pd.to_numeric(raw_reference, errors="coerce")
    reference_is_numeric = numeric_reference.notna().all()

    if reference_is_numeric:
        reference_value = numeric_reference.astype(float)
        if "sample_id" in raw_df.columns:
            sample_labels = raw_df.loc[valid_index, "sample_id"].astype(str).fillna("sample").reset_index(drop=True)
        elif "id" in raw_df.columns:
            sample_labels = raw_df.loc[valid_index, "id"].astype(str).fillna("sample").reset_index(drop=True)
        else:
            sample_labels = pd.Series([f"sample {ix}" for ix in range(len(valid_index))])
        hover_text = [
            f"sample={label}<br>{reference_col}={value:.3g}"
            for label, value in zip(sample_labels, reference_value)
        ]
        out["label"] = sample_labels
        out["manifold_position"] = reference_value
        out["reference_value"] = reference_value
    else:
        reference_value = raw_reference.astype(str).fillna("missing")
        hover_text = [f"{reference_col}={value}" for value in reference_value]
        out["label"] = reference_value
        out["manifold_position"] = pd.factorize(reference_value)[0]
        out["reference_value"] = reference_value

    out["hover_text"] = hover_text
    out["reference_is_numeric"] = reference_is_numeric
    out["has_reference"] = True
    out["reference_name"] = reference_col
    return out


def embed_data(x: np.ndarray, method: str, n_neighbors: int, perplexity: int, seed: int, n_components: int) -> np.ndarray:
    if method == "PCA":
        return _standardize_embedding(PCA(n_components=n_components, random_state=seed).fit_transform(x))
    if method == "Isomap":
        return _standardize_embedding(Isomap(n_neighbors=n_neighbors, n_components=n_components).fit_transform(x))
    if method == "UMAP":
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP requires the umap-learn package. Install it with `pip install umap-learn`.")
        return _standardize_embedding(
            umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                random_state=seed,
            ).fit_transform(x)
        )
    return _standardize_embedding(
        TSNE(
            n_components=n_components,
            perplexity=min(perplexity, max(5, (x.shape[0] - 1) // 3)),
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(x)
    )


def bind_metric_for_app(embedding: np.ndarray, h_vectors: np.ndarray, h_values: np.ndarray) -> pd.DataFrame:
    n_components = embedding.shape[1]
    embedding_df = pd.DataFrame(embedding, columns=[f"embedding_{ix}" for ix in range(n_components)])
    vectors_df = pd.DataFrame([vectors.flatten() for vectors in h_vectors])
    values_df = pd.DataFrame(h_values)
    out = pd.concat([embedding_df.reset_index(drop=True), vectors_df, values_df], axis=1)
    if n_components == 2:
        metric_columns = ["x0", "x1", "y0", "y1", "s0", "s1"]
    else:
        metric_columns = [f"v{i}_{j}" for i in range(n_components) for j in range(n_components)]
        metric_columns += [f"s{i}" for i in range(n_components)]
    out.columns = list(embedding_df.columns) + metric_columns
    if n_components == 2:
        out["angle"] = np.arctan(out["y1"] / out["x1"]) * (180 / np.pi)
    return out


def compute_local_distortions(
    x: np.ndarray,
    y: np.ndarray,
    n_neighbors: int,
    affinity_radius: float,
) -> pd.DataFrame:
    geom = Geometry(
        adjacency_kwds={"n_neighbors": n_neighbors},
        affinity_kwds={"radius": affinity_radius},
    )
    geom.set_data_matrix(x)
    laplacian = geom.compute_laplacian_matrix()
    _, _, h_vectors, h_values, _, _ = riemann_metric(y, laplacian, n_dim=y.shape[1])
    out = bind_metric_for_app(y, h_vectors, h_values)
    s_cols = [f"s{i}" for i in range(y.shape[1])]
    out["distortion_ratio"] = np.divide(
        out[s_cols].max(axis=1),
        np.maximum(out[s_cols].min(axis=1), 1e-8),
    )
    out["local_area"] = np.maximum(out[s_cols], 0).prod(axis=1) ** (1 / len(s_cols))
    return out.replace([np.inf, -np.inf], np.nan).fillna(0)


def neighborhood_distances(x: np.ndarray, y: np.ndarray, n_neighbors: int) -> pd.DataFrame:
    model = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(x)
    true_dist, idx = model.kneighbors(x)
    centers = np.repeat(np.arange(x.shape[0]), n_neighbors)
    neighbors = idx[:, 1:].reshape(-1)
    true = true_dist[:, 1:].reshape(-1)
    embedding = np.linalg.norm(y[centers] - y[neighbors], axis=1)
    return pd.DataFrame({"center": centers, "neighbor": neighbors, "true": true, "embedding": embedding})


def broken_links(dists: pd.DataFrame, n_bins: int, outlier_factor: float) -> pd.DataFrame:
    edges = dists.copy()
    edges["bin"] = pd.cut(edges["true"], bins=n_bins, duplicates="drop")
    edges["broken"] = False
    for _, ix in edges.groupby("bin", observed=False).groups.items():
        values = edges.loc[ix, "embedding"]
        if len(values) < 5:
            continue
        q1, q3 = np.percentile(values, [25, 75])
        threshold = q3 + outlier_factor * (q3 - q1)
        edges.loc[ix, "broken"] = values > threshold
    return edges


@st.cache_data(show_spinner=False)
def run_pipeline(
    x: np.ndarray,
    meta: pd.DataFrame,
    embedding_method: str,
    n_components: int,
    n_neighbors: int,
    affinity_radius: float,
    perplexity: int,
    outlier_factor: float,
    seed: int,
    provided_embedding: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_components = provided_embedding.shape[1] if provided_embedding is not None else n_components
    y = provided_embedding if provided_embedding is not None else embed_data(x, embedding_method, n_neighbors, perplexity, seed, n_components)
    metrics = compute_local_distortions(x, y, n_neighbors, affinity_radius)
    dists = neighborhood_distances(x, y, n_neighbors)
    links = broken_links(dists, n_bins=10, outlier_factor=outlier_factor)
    metrics = pd.concat([metrics, meta.reset_index(drop=True)], axis=1)
    return metrics, dists, links


def ellipse_points(row: pd.Series, scale: float, resolution: int = 36) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0, 2 * np.pi, resolution)
    axes = np.sqrt(np.maximum([row["s1"], row["s0"]], 1e-8))
    axes = axes / np.nanmedian(axes) * scale
    basis = np.array([[row["x0"], row["x1"]], [row["y0"], row["y1"]]], dtype=float)
    circle = np.vstack([axes[0] * np.cos(theta), axes[1] * np.sin(theta)])
    ellipse = basis @ circle
    return row["embedding_0"] + ellipse[0], row["embedding_1"] + ellipse[1]


def ellipsoid_points(row: pd.Series, scale: float, resolution: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones_like(u), np.cos(v))
    sphere = np.stack([sphere_x, sphere_y, sphere_z], axis=0).reshape(3, -1)

    axes = np.sqrt(np.maximum([row["s0"], row["s1"], row["s2"]], 1e-8))
    axes = axes / np.nanmedian(axes) * scale
    basis = np.array(
        [
            [row["v0_0"], row["v0_1"], row["v0_2"]],
            [row["v1_0"], row["v1_1"], row["v1_2"]],
            [row["v2_0"], row["v2_1"], row["v2_2"]],
        ],
        dtype=float,
    )
    ellipsoid = basis.T @ np.diag(axes) @ sphere
    ellipsoid[0] += row["embedding_0"]
    ellipsoid[1] += row["embedding_1"]
    ellipsoid[2] += row["embedding_2"]
    shape = sphere_x.shape
    return ellipsoid[0].reshape(shape), ellipsoid[1].reshape(shape), ellipsoid[2].reshape(shape)


def ellipsoid_axes(row: pd.Series, scale: float) -> list[tuple[list[float], list[float], list[float]]]:
    axes = np.sqrt(np.maximum([row["s0"], row["s1"], row["s2"]], 1e-8))
    axes = axes / np.nanmedian(axes) * scale
    basis = np.array(
        [
            [row["v0_0"], row["v0_1"], row["v0_2"]],
            [row["v1_0"], row["v1_1"], row["v1_2"]],
            [row["v2_0"], row["v2_1"], row["v2_2"]],
        ],
        dtype=float,
    )
    center = np.array([row["embedding_0"], row["embedding_1"], row["embedding_2"]], dtype=float)
    segments = []
    for ix in range(3):
        delta = basis[ix] * axes[ix]
        start = center - delta
        end = center + delta
        segments.append(([start[0], end[0]], [start[1], end[1]], [start[2], end[2]]))
    return segments


def make_plot(
    df: pd.DataFrame,
    links: pd.DataFrame,
    color_by: str,
    ellipse_stride: int,
    ellipse_scale: float,
    show_links: bool,
    show_ellipsoids: bool = False,
    max_ellipsoids: int = 45,
) -> go.Figure:
    is_reference_color = color_by == "reference"
    reference_is_numeric = bool(df["reference_is_numeric"].iloc[0]) if "reference_is_numeric" in df else True
    if is_reference_color:
        colors = df["reference_value"] if reference_is_numeric else pd.factorize(df["reference_value"].astype(str))[0]
    else:
        colors = df[color_by]
    fig = go.Figure()
    is_3d = "embedding_2" in df.columns

    if show_links and not is_3d:
        broken = links[links["broken"]].head(180)
        for edge in broken.itertuples(index=False):
            a = df.iloc[int(edge.center)]
            b = df.iloc[int(edge.neighbor)]
            fig.add_trace(
                go.Scatter(
                    x=[a.embedding_0, b.embedding_0],
                    y=[a.embedding_1, b.embedding_1],
                    mode="lines",
                    line={"color": "rgba(228, 87, 46, 0.18)", "width": 1},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    if not is_3d and show_ellipsoids:
        for _, row in df.iloc[::ellipse_stride].iterrows():
            ex, ey = ellipse_points(row, ellipse_scale)
            fig.add_trace(
                go.Scatter(
                    x=ex,
                    y=ey,
                    mode="lines",
                    line={"color": "rgba(41, 49, 51, 0.34)", "width": 1},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    if is_3d and show_ellipsoids and max_ellipsoids > 0:
        ellipsoid_count = min(max_ellipsoids, len(df))
        ellipsoid_indices = np.unique(np.linspace(0, len(df) - 1, ellipsoid_count, dtype=int))
        for _, row in df.iloc[ellipsoid_indices].iterrows():
            ex, ey, ez = ellipsoid_points(row, ellipse_scale)
            fig.add_trace(
                go.Surface(
                    x=ex,
                    y=ey,
                    z=ez,
                    surfacecolor=np.zeros_like(ex),
                    colorscale=[[0, "#aeb8c2"], [1, "#aeb8c2"]],
                    opacity=0.34,
                    showscale=False,
                    hoverinfo="skip",
                    name="metric ellipsoid",
                    showlegend=False,
                )
            )
            for ax, ay, az in ellipsoid_axes(row, ellipse_scale):
                fig.add_trace(
                    go.Scatter3d(
                        x=ax,
                        y=ay,
                        z=az,
                        mode="lines",
                        line={"color": "rgba(31, 41, 55, 0.62)", "width": 3},
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    if show_links and is_3d:
        broken = links[links["broken"]].head(180)
        for edge in broken.itertuples(index=False):
            a = df.iloc[int(edge.center)]
            b = df.iloc[int(edge.neighbor)]
            fig.add_trace(
                go.Scatter3d(
                    x=[a.embedding_0, b.embedding_0],
                    y=[a.embedding_1, b.embedding_1],
                    z=[a.embedding_2, b.embedding_2],
                    mode="lines",
                    line={"color": "rgba(228, 87, 46, 0.78)", "width": 5},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    marker = {
        "size": 7,
        "color": colors,
        "line": {"width": 0.5, "color": "rgba(255,255,255,0.7)"},
    }
    if is_reference_color and not reference_is_numeric:
        marker["colorscale"] = "Plotly3"
        marker["showscale"] = False
    else:
        marker["colorscale"] = "Viridis"
        marker["showscale"] = True
    if is_3d:
        fig.add_trace(
            go.Scatter3d(
                x=df["embedding_0"],
                y=df["embedding_1"],
                z=df["embedding_2"],
                mode="markers",
                marker=marker,
                text=df["hover_text"] if "hover_text" in df else df["label"],
                customdata=np.stack([df["distortion_ratio"], df["local_area"]], axis=1),
                hovertemplate=(
                    "%{text}<br>"
                    "axis ratio=%{customdata[0]:.2f}<br>"
                    "local scale=%{customdata[1]:.2f}<extra></extra>"
                ),
                name="samples",
                showlegend=False,
            )
        )
        fig.update_layout(
            height=690,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            paper_bgcolor="white",
            showlegend=False,
            scene={
                "xaxis": {"title": "", "showticklabels": False},
                "yaxis": {"title": "", "showticklabels": False},
                "zaxis": {"title": "", "showticklabels": False},
                "aspectmode": "data",
            },
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df["embedding_0"],
                y=df["embedding_1"],
                mode="markers",
                marker=marker,
                text=df["hover_text"] if "hover_text" in df else df["label"],
                customdata=np.stack([df["distortion_ratio"], df["local_area"]], axis=1),
                hovertemplate=(
                    "%{text}<br>"
                    "axis ratio=%{customdata[0]:.2f}<br>"
                    "local area=%{customdata[1]:.2f}<extra></extra>"
                ),
                name="samples",
                showlegend=False,
            )
        )
        fig.update_layout(
            height=690,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            paper_bgcolor="white",
            plot_bgcolor="#fbfaf7",
            showlegend=False,
            xaxis={"title": "", "showgrid": False, "zeroline": False, "showticklabels": False},
            yaxis={"title": "", "showgrid": False, "zeroline": False, "showticklabels": False, "scaleanchor": "x", "scaleratio": 1},
        )
    return fig


def render_downloads_and_citations() -> None:
    st.subheader("Run locally")
    st.write(
        "For larger private datasets, download a portable version and run the app locally. "
        "No Python package installation is needed after unzipping."
    )
    st.markdown("**Windows**")
    st.link_button("Download Windows zip", f"{RELEASE_BASE_URL}/DistortionsDemo_Windows.zip")
    st.caption("After downloading, unzip the file and run `Start Distortions Demo.bat`.")

    st.markdown("**macOS**")
    st.link_button("Download macOS zip", f"{RELEASE_BASE_URL}/DistortionsDemo_macOS.zip")
    st.caption("After downloading, unzip the file and open `Start Distortions Demo.command`.")
    with st.expander("macOS security warning help", expanded=False):
        st.markdown(
            """
            Because this is a portable demo app and is not Apple-notarized yet, macOS may show a
            security warning the first time you open it.

            Try this first:

            1. Unzip the downloaded file.
            2. Right-click or Control-click `Start Distortions Demo.command`.
            3. Choose **Open**.
            4. If macOS asks again, choose **Open** once more.

            If macOS still blocks it, open Terminal and run:

            ```bash
            cd ~/Downloads
            xattr -dr com.apple.quarantine DistortionsDemo
            ```

            Then open `Start Distortions Demo.command` again.

            What this command does:

            - `cd ~/Downloads` moves Terminal to the Downloads folder, assuming the app was unzipped there.
            - `xattr` edits extended file attributes on macOS.
            - `-d com.apple.quarantine` removes the quarantine flag macOS adds to files downloaded from the internet.
            - `-r` applies this change recursively to the whole `DistortionsDemo` folder.

            This does not disable macOS security globally. It only removes the download quarantine
            flag for this demo folder.
            """
        )

    st.subheader("Citations")
    st.markdown(
        f"""
        **distortions package paper**  
        Sankaran, Zhang, Chenab, and Meila. *Interactive visualization of metric distortion in nonlinear data embeddings using the distortions package.* Briefings in Bioinformatics, 2026. [Paper]({DISTORTIONS_PAPER_URL})

        **RMetric method paper**  
        Perraul-Joncas and Meila. *Non-linear dimensionality reduction: Riemannian metric estimation and the problem of geometric discovery.* arXiv:1305.7255, 2013. [Paper]({RMETRIC_PAPER_URL})

        **Software**  
        [distortions GitHub repository]({DISTORTIONS_PACKAGE_URL})  
        [distortions documentation]({DISTORTIONS_DOCS_URL})
        """
    )


def render_raw_upload_guide(n_components: int) -> None:
    st.subheader("Upload raw data")
    st.write(
        "Use this workflow when you have original measurements but do not already have an embedding. "
        f"The app will compute a {n_components}D embedding, then evaluate local distortion."
    )

    st.markdown("**Example: raw Swiss roll data**")
    st.dataframe(example_preview(str(RAW_DATA_EXAMPLE_PATH)), width="stretch")
    st.download_button(
        "Download synthetic raw data example",
        data=example_csv_bytes(str(RAW_DATA_EXAMPLE_PATH)),
        file_name="raw_swiss_roll_example.csv",
        mime="text/csv",
        help="Download a 240-row synthetic Swiss roll table showing the expected raw data format.",
    )

    st.markdown(
        """
        **1. Prepare your table**

        Each row should be one sample. In the Swiss roll example above, `sample_000`,
        `sample_001`, ... are individual samples.

        The columns you want to embed should be numeric measurement columns. In this example,
        `feature_1`, `feature_2`, and `feature_3` are the original Swiss roll coordinates. These
        are the columns the app uses to compute PCA, Isomap, t-SNE, or UMAP, and they are also used
        to estimate distortion in the original data space.

        Feature names are flexible. They do not have to be called `feature_1`, `feature_2`, etc.
        For larger datasets, a helpful naming pattern is `prefix_1`, `prefix_2`, ... or
        `prefix_01`, `prefix_02`, ... . The prefix can be anything, such as `gene`, `protein`,
        `image`, `pixel`, or `feature`. The app detects columns with the same prefix as a feature
        group, so users can choose one group instead of manually selecting many columns.

        **2. Add optional columns for interpretation**

        Extra columns can help explain the plot, but they are not used to compute the embedding.
        In the example, `sample_id` is an ID column. `label` is a categorical reference column, so it
        can be used for hover text or discrete color groups. `manifold_position` is a numeric
        reference column, so it can be used for hover text or continuous coloring.

        These optional column names are also flexible. Your file can use names such as `cell_type`,
        `batch`, `time`, `patient_group`, or any other metadata that helps interpret the embedding.

        **3. Upload and run the analysis**

        After uploading, the app suggests feature columns automatically. If it detects a feature
        group like `gene_1`, `gene_2`, ... or `feature_1`, `feature_2`, ... , you can use the whole
        group, choose an index range, or exclude specific columns. If your columns do not follow a
        shared prefix pattern, switch to custom selection and choose the numeric columns manually.

        When the setup looks right, click **Run analysis**. After the first run, you can switch
        between PCA, Isomap, t-SNE, and UMAP and adjust visualization options without uploading the
        data again.
        """
    )


def render_embedding_upload_guide(n_components: int) -> None:
    st.subheader("Evaluate my embedding")
    st.write(
        f"Use this workflow when you already computed a {n_components}D embedding elsewhere. "
        "The app will keep your uploaded coordinates fixed and evaluate their local distortion."
    )

    st.markdown("**Example: Swiss roll data with an existing embedding**")
    st.dataframe(example_preview(str(EMBEDDING_DATA_EXAMPLE_PATH)), width="stretch")
    st.download_button(
        "Download synthetic embedding example",
        data=example_csv_bytes(str(EMBEDDING_DATA_EXAMPLE_PATH)),
        file_name="existing_embedding_swiss_roll_example.csv",
        mime="text/csv",
        help="Download a 240-row synthetic Swiss roll table with raw features plus existing Isomap coordinates.",
    )

    st.markdown(
        f"""
        **1. Include the original feature columns**

        The app still needs the original measurements, even when you upload your own embedding.
        These columns define the original data space used for local geometry and distortion
        estimation. In the example above, `feature_1`, `feature_2`, and `feature_3` are the raw
        Swiss roll coordinates.

        Feature names are flexible. They do not have to be called `feature_1`, `feature_2`, etc.
        For many columns, use a shared prefix such as `gene_1`, `gene_2`, ... or `protein_01`,
        `protein_02`, ... so the app can detect a feature group. If your columns do not follow a
        shared prefix pattern, use custom selection.

        **2. Include your embedding coordinate columns**

        Add `{n_components}` numeric coordinate columns for the embedding you want to evaluate.
        Names like `embedding_0`, `embedding_1`{", and `embedding_2`" if n_components == 3 else ""}
        are detected automatically, but the names can be different. If needed, select the coordinate
        columns manually in the sidebar.

        The app also recognizes common method-specific names such as `isomap_1`, `isomap_2`,
        `umap_1`, `umap_2`, `tsne_1`, `tsne_2`, or `pca_1`, `pca_2`. When these columns are detected
        as embedding coordinates, they are automatically excluded from the original feature columns.
        For example, in the table above, `feature_1`, `feature_2`, and `feature_3` define the
        original geometry, while `isomap_1` and `isomap_2` define the fixed 2D layout being
        evaluated.

        These embedding coordinate columns are the layout being evaluated. They are not used as
        original features, and the app will not recompute them. Original features and embedding
        coordinates are matched row by row. If you combine them from separate files, make sure the
        sample order is the same before uploading.

        **3. Upload and run the evaluation**

        After uploading, check that the app selected the correct original feature columns and the
        correct embedding coordinate columns. Optional columns such as `label`, `cell_type`, `time`,
        or `batch` can be used for hover text and coloring, but they are not used for distortion
        estimation.

        When the setup looks right, click **Run analysis**. The app will keep your uploaded
        coordinates fixed and estimate distortion for that embedding.
        """
    )


def read_upload_or_stop(uploaded_file) -> pd.DataFrame:
    try:
        return read_uploaded_table(uploaded_file.getvalue(), uploaded_file.name)
    except Exception as exc:
        st.error(f"Could not read uploaded file: {exc}")
        st.stop()


def looks_like_metadata(col: str) -> bool:
    name = col.lower()
    return (
        name in {
            "label",
            "class",
            "group",
            "sample_id",
            "id",
            "manifold_position",
            "sample_index",
            "distortion_ratio",
            "local_area",
            "s0",
            "s1",
            "x0",
            "x1",
            "y0",
            "y1",
        }
        or name.endswith("_id")
        or name.startswith("label")
    )


def find_embedding_defaults(numeric_cols: list[str], n_components: int = 2) -> list[str]:
    lower_to_col = {col.lower(): col for col in numeric_cols}
    bases = ["embedding", "umap", "isomap", "tsne", "t_sne", "pca"]
    starts = [0, 1]
    for base in bases:
        for start in starts:
            names = [f"{base}_{ix}" for ix in range(start, start + n_components)]
            if all(name in lower_to_col for name in names):
                return [lower_to_col[name] for name in names]
    return []


def detect_feature_groups(numeric_cols: list[str], excluded_cols: set[str]) -> dict[str, list[str]]:
    groups: dict[str, list[tuple[int, str]]] = {}
    embedding_prefixes = {"embedding", "umap", "isomap", "tsne", "t_sne", "pca"}
    for col in numeric_cols:
        if col in excluded_cols or looks_like_metadata(col):
            continue
        match = re.match(r"^(.+)_([0-9]+)$", col)
        if match is None:
            continue
        prefix = match.group(1)
        if prefix.lower() in embedding_prefixes:
            continue
        groups.setdefault(prefix, []).append((int(match.group(2)), col))

    return {
        prefix: [col for _, col in sorted(items)]
        for prefix, items in sorted(groups.items())
        if len(items) >= 2
    }


def feature_suffix_number(col: str) -> int | None:
    match = re.match(r"^.+_([0-9]+)$", col)
    return int(match.group(1)) if match is not None else None


def default_label_column(raw_df: pd.DataFrame) -> str:
    for candidate in ["manifold_position", "label", "class", "group", "sample_id"]:
        if candidate in raw_df.columns:
            return candidate
    return "None"


def select_uploaded_columns(raw_df: pd.DataFrame, require_embedding: bool, n_components: int) -> tuple[list[str], str | None, list[str] | None, int]:
    numeric_cols = raw_df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("The uploaded table needs at least two numeric columns.")
        st.stop()

    embedding_cols = None
    if require_embedding:
        embedding_defaults = find_embedding_defaults(numeric_cols, n_components)
        if not embedding_defaults:
            st.warning(
                f"This file does not appear to contain obvious {n_components}D embedding columns. "
                "If you only have raw features, use the Upload raw data tab instead."
            )
        embedding_cols = st.multiselect(
            "Existing embedding columns",
            numeric_cols,
            default=embedding_defaults,
            max_selections=n_components,
            help=f"Select exactly {n_components} numeric columns containing the existing {n_components}D embedding coordinates.",
        )
        st.caption(f"These {n_components} columns are the {n_components}D layout you want to evaluate. They are not used as original features.")
        if len(embedding_cols) != n_components:
            st.error(f"Select exactly {n_components} embedding columns.")
            st.stop()

    likely_embedding_cols = set(find_embedding_defaults(numeric_cols, 2) + find_embedding_defaults(numeric_cols, 3))
    default_features = [
        col for col in numeric_cols
        if col not in (embedding_cols or []) and col not in likely_embedding_cols and not looks_like_metadata(col)
    ]
    feature_groups = detect_feature_groups(numeric_cols, set(embedding_cols or []) | likely_embedding_cols)
    group_options = list(feature_groups) + ["Auto-detected numeric columns", "Custom selection"]
    default_group = max(feature_groups, key=lambda key: len(feature_groups[key])) if feature_groups else "Auto-detected numeric columns"
    feature_group = default_group
    if feature_group in feature_groups:
        feature_cols = feature_groups[feature_group]
    elif feature_group == "Auto-detected numeric columns":
        feature_cols = default_features
    else:
        feature_cols = default_features
    default_label = default_label_column(raw_df)
    label_choice = default_label if default_label in raw_df.columns else "None"
    row_cap = min(len(raw_df), 1500)
    default_rows = min(len(raw_df), 400)
    max_rows = default_rows

    with st.expander("Uploaded data options", expanded=True):
        if feature_groups:
            st.caption(
                "Detected feature groups: "
                + ", ".join(f"`{prefix}` ({len(cols)} columns)" for prefix, cols in feature_groups.items())
            )
        else:
            st.caption("No `prefix_01`, `prefix_02`, ... feature groups were detected.")
        feature_group = st.selectbox(
            "Feature group",
            group_options,
            index=group_options.index(default_group),
            help=(
                "Choose which numeric column group should be embedded. Columns named like "
                "`gene_01`, `gene_02`, ... are grouped as `gene`."
            ),
        )
        if feature_group in feature_groups:
            feature_cols = feature_groups[feature_group]
        elif feature_group == "Auto-detected numeric columns":
            feature_cols = default_features
        else:
            feature_cols = default_features

        st.caption(
            "Feature columns define the original data space used for embedding and distortion estimation. "
            "For grouped columns, use all features by default, choose an index range, or exclude a few columns."
        )
        if feature_group in feature_groups:
            numbered = [(feature_suffix_number(col), col) for col in feature_cols]
            numbered = [(num, col) for num, col in numbered if num is not None]
            if numbered:
                numbers = [num for num, _ in numbered]
                min_ix, max_ix = min(numbers), max(numbers)
                c1, c2 = st.columns(2)
                with c1:
                    start_ix = st.number_input("Start index", min_value=min_ix, max_value=max_ix, value=min_ix)
                with c2:
                    end_ix = st.number_input("End index", min_value=min_ix, max_value=max_ix, value=max_ix)
                if start_ix > end_ix:
                    st.error("Start index must be less than or equal to end index.")
                    st.stop()
                feature_cols = [col for num, col in numbered if start_ix <= num <= end_ix]
            st.caption(
                f"Using {len(feature_cols)} `{feature_group}_*` columns"
                + (f", from `{feature_cols[0]}` to `{feature_cols[-1]}`." if feature_cols else ".")
            )
            excluded_cols = st.multiselect(
                "Exclude columns",
                feature_cols,
                default=[],
                help="Optional. Remove a small number of columns from the selected feature group.",
            )
            feature_cols = [col for col in feature_cols if col not in excluded_cols]
            st.caption(f"Final feature count: {len(feature_cols)}.")
        elif feature_group == "Auto-detected numeric columns":
            st.caption(f"Using all {len(feature_cols)} auto-detected numeric feature columns.")
            excluded_cols = st.multiselect(
                "Exclude columns",
                feature_cols,
                default=[],
                help="Optional. Remove a small number of auto-detected columns.",
            )
            feature_cols = [col for col in feature_cols if col not in excluded_cols]
            st.caption(f"Final feature count: {len(feature_cols)}.")
        else:
            feature_cols = st.multiselect(
                "Feature columns to embed",
                numeric_cols,
                default=feature_cols,
                help="Numeric columns used as the original high-dimensional data matrix.",
            )
            st.caption("Choose the exact numeric columns to use as the original data space.")

        if row_cap > 50:
            max_rows = st.slider(
                "Rows to analyze",
                50,
                row_cap,
                default_rows,
                step=50,
                help="Maximum number of rows used for the analysis. Subsampling keeps the web app responsive for large files.",
            )
        else:
            st.caption(f"Using all {row_cap} rows.")
            max_rows = row_cap

    feature_cols = [col for col in feature_cols if col not in (embedding_cols or [])]
    if len(feature_cols) < 2:
        st.error("Select at least two feature columns. Embedding coordinate columns should not be used as features.")
        st.stop()
    label_col = None if label_choice == "None" else label_choice
    st.caption(f"Features: {len(feature_cols)} numeric columns.")

    return feature_cols, label_col, embedding_cols, max_rows


def render_results(
    metrics: pd.DataFrame,
    links: pd.DataFrame,
    color_by: str,
    ellipse_stride: int,
    ellipse_scale: float,
    show_links: bool,
    show_ellipsoids: bool = False,
    max_ellipsoids: int = 45,
    feature_export: pd.DataFrame | None = None,
) -> None:
    left, right = st.columns([0.72, 0.28], gap="large")
    is_3d = "embedding_2" in metrics.columns

    with left:
        st.plotly_chart(
            make_plot(metrics, links, color_by, ellipse_stride, ellipse_scale, show_links, show_ellipsoids, max_ellipsoids),
            width="stretch",
            config={"displayModeBar": True, "scrollZoom": True},
        )

    with right:
        broken_count = int(links["broken"].sum())
        st.metric("Flagged neighbor links", f"{broken_count:,}", f"{broken_count / max(len(links), 1):.1%}")
        st.caption("Original-space nearest-neighbor pairs whose embedded distance is unusually large.")
        st.metric("Median axis ratio", f"{metrics['distortion_ratio'].median():.2f}")
        st.caption("Typical local anisotropy. Larger values mean the local metric ellipse is more elongated.")
        st.metric("95th pct axis ratio", f"{metrics['distortion_ratio'].quantile(0.95):.2f}")
        st.caption("Upper-tail local anisotropy. This highlights severe distortion among the most stretched samples.")

    with st.expander("How to read the visualization", expanded=True):
        if is_3d:
            st.write(
                "Each point is one high-dimensional sample after embedding into 3D. Color summarizes local "
                "metric distortion estimated from the original space. Orange links mark original-space "
                "neighbors that are unusually far apart in the 3D embedding. Optional ellipsoids summarize "
                "the local 3D Riemannian metric; elongated ellipsoids indicate directions that are stretched unevenly."
            )
        else:
            st.write(
                "Each point is one high-dimensional sample after embedding. The ellipses summarize the local "
                "Riemannian metric estimated from the original space: elongated ellipses indicate directions "
                "that the embedding stretches unevenly. Orange links mark original-space neighbors that are "
                "unusually far apart in the embedding."
            )

    st.download_button(
        "Download embedding and metrics CSV",
        data=results_csv(metrics, feature_export),
        file_name="distortion_embedding_metrics.csv",
        mime="text/csv",
        help="Download feature columns, plotted embedding coordinates, and distortion metrics.",
        width="stretch",
    )
    if is_3d:
        st.caption("This CSV can be re-uploaded in the Evaluate my embedding tab using `embedding_0`, `embedding_1`, and `embedding_2` as embedding columns.")
    else:
        st.caption("This CSV can be re-uploaded in the Evaluate my embedding tab using `embedding_0` and `embedding_1` as embedding columns.")

    st.subheader("Most distorted samples")
    embedding_cols = [col for col in ["embedding_0", "embedding_1", "embedding_2"] if col in metrics.columns]
    st.dataframe(
        metrics[embedding_cols + ["label", "distortion_ratio", "local_area"]]
        .sort_values("distortion_ratio", ascending=False)
        .head(12),
        width="stretch",
    )


st.title("Visualizing Distortions in Low-Dimensional Embeddings")
st.caption("Interactive companion demo for explaining local metric distortion and broken neighborhoods.")

stage = st.segmented_control(
    "Workflow",
    ["Demo", "Upload raw data", "Evaluate my embedding", "Downloads & citations"],
    default="Demo",
    label_visibility="collapsed",
)

if stage == "Downloads & citations":
    render_downloads_and_citations()
    st.stop()

with st.sidebar:
    st.subheader(stage)
    seed = 7
    embedding_dimension = st.segmented_control(
        "Embedding dimension",
        ["2D", "3D"],
        default="2D",
        help=(
            "Choose the dimensionality of the embedding. In 2D, local metrics are shown as ellipses. "
            "In 3D, the app shows a 3D scatter plot with distortion coloring and broken links."
        ),
    )
    n_components = 3 if embedding_dimension == "3D" else 2

    provided_embedding = None
    feature_export = None
    raw_df = None
    valid_index = None
    default_reference_col = None
    run_analysis = stage == "Demo"
    if stage == "Demo":
        dataset = st.selectbox(
            "Dataset",
            list(DATASET_HELP),
            help="Synthetic example used for the talk demo. Swiss roll and S-curve have known manifold structure; Two clusters shows cluster-level distortion.",
        )
        n_samples = 240
        noise = 0.08
        x, meta = make_dataset(dataset, n_samples, noise, seed)
        feature_export = pd.DataFrame(
            x,
            columns=[f"feature_{ix + 1}" for ix in range(x.shape[1])],
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload data",
            type=["csv", "tsv", "txt", "xlsx", "xls"],
            help="Upload a table file. Supported formats: CSV, TSV/TXT, XLSX, and XLS.",
        )
        if uploaded_file is None:
            x = None
            meta = None
            st.session_state.pop("uploaded_run_key", None)
        else:
            raw_df = read_upload_or_stop(uploaded_file)
            with st.expander("Preview uploaded table", expanded=False):
                st.dataframe(raw_df.head(8), width="stretch")
            feature_cols, default_reference_col, embedding_cols, max_rows = select_uploaded_columns(
                raw_df,
                require_embedding=(stage == "Evaluate my embedding"),
                n_components=n_components,
            )
            x, meta, provided_embedding, feature_export, valid_index = prepare_uploaded_dataset(
                raw_df,
                feature_cols,
                None,
                embedding_cols,
                max_rows,
                seed,
            )
            if x.shape[0] < 10:
                st.error("After dropping missing values, at least 10 rows are needed for this demo.")
                st.stop()
            st.caption(f"Using {x.shape[0]} rows and {x.shape[1]} numeric features.")
            upload_run_key = (
                stage,
                n_components,
                uploaded_file.name,
                uploaded_file.size,
                tuple(feature_cols),
                tuple(embedding_cols or []),
                max_rows,
            )
            if st.button("Run analysis", type="primary", width="stretch"):
                st.session_state["uploaded_run_key"] = upload_run_key
            run_analysis = st.session_state.get("uploaded_run_key") == upload_run_key
            if not run_analysis:
                st.info("Review the uploaded data options, then click Run analysis.")
                x = None
                meta = None

    if x is not None:
        st.subheader("Embedding")
        if provided_embedding is None:
            embedding_options = ["PCA", "Isomap", "t-SNE", "UMAP"]
            embedding_method = st.segmented_control(
                "Embedding",
                embedding_options,
                default="Isomap",
                help="Method used to compute the 2D embedding shown in the plot.",
            )
            if embedding_method == "UMAP" and not UMAP_AVAILABLE:
                st.warning("UMAP is not installed in this environment. Install `umap-learn` to enable this option.")
        else:
            embedding_method = "Uploaded"
            st.info("Using the selected embedding columns from the uploaded file.")
        max_neighbors = max(2, min(40, x.shape[0] - 1))
        min_neighbors = min(6, max_neighbors)
        default_neighbors = min(14, max_neighbors)
        perplexity = 30
        n_neighbors = default_neighbors
        affinity_radius = 1.6
        outlier_factor = 1.5

        st.subheader("Visualization")
        if raw_df is not None and valid_index is not None:
            reference_options = ["None"] + raw_df.columns.tolist()
            reference_default = default_reference_col if default_reference_col in raw_df.columns else "None"
            reference_choice = st.selectbox(
                "Hover/color column",
                reference_options,
                index=reference_options.index(reference_default),
                help="Optional uploaded column used only for hover text and the Color by option below.",
            )
            meta = apply_reference_column(
                meta,
                raw_df,
                valid_index,
                None if reference_choice == "None" else reference_choice,
            )
        color_options = ["distortion_ratio", "local_area"]
        if "has_reference" in meta and bool(meta["has_reference"].iloc[0]):
            color_options.append("reference")
        reference_name = str(meta["reference_name"].iloc[0]) if "reference_name" in meta else "selected column"
        color_by = st.selectbox(
            "Color by",
            color_options,
            format_func=lambda key: reference_name if key == "reference" else COLOR_LABELS[key],
            help="Choose what point color represents in the embedding plot.",
        )
        if color_by == "reference":
            st.caption(f"Color shows `{reference_name}`. Numeric columns use a continuous scale; categorical columns use distinct colors.")
        else:
            st.caption(COLOR_HELP[color_by])
        if n_components == 2:
            show_ellipsoids = True
            ellipse_stride = 5
            ellipse_scale = 0.05
            max_ellipsoids = 45
        else:
            show_ellipsoids = True
            st.caption(
                "3D mode uses point color and broken-neighborhood links to show distortion. "
                "Ellipsoids show the local 3D metric, but can add visual clutter."
            )
            ellipse_stride = 1
            ellipse_scale = 0.12
            max_ellipsoids = 24
        show_links = True
        with st.expander("Advanced controls", expanded=False):
            if embedding_method == "t-SNE":
                max_perplexity = max(5, min(80, (x.shape[0] - 1) // 3))
                perplexity = st.slider(
                    "t-SNE perplexity",
                    5,
                    max_perplexity,
                    min(30, max_perplexity),
                    help="t-SNE neighborhood scale. Higher values make t-SNE consider broader neighborhoods.",
                )
            n_neighbors = st.slider(
                "Local neighborhood size",
                min_neighbors,
                max_neighbors,
                default_neighbors,
                help=(
                    "Number of nearest neighbors in the original data used to estimate local geometry "
                    "and flag broken neighbor links. For Isomap and UMAP, this same value also controls "
                    "the embedding neighborhood size."
                ),
            )
            affinity_radius = st.slider(
                "Affinity radius",
                0.2,
                5.0,
                affinity_radius,
                step=0.1,
                help="Radius for the Gaussian affinity kernel used when constructing the original-space graph Laplacian.",
            )
            outlier_factor = st.slider(
                "Broken-link sensitivity",
                0.5,
                4.0,
                outlier_factor,
                step=0.1,
                help="Controls the boxplot-style outlier threshold for flagged neighbor links. Smaller values flag more links.",
            )
            show_ellipsoids = st.toggle(
                "Show metric glyphs",
                value=show_ellipsoids,
                help="Draw ellipses or ellipsoids summarizing the local Riemannian metric.",
            )
            if show_ellipsoids and n_components == 2:
                ellipse_stride = st.slider(
                    "Ellipse density",
                    1,
                    12,
                    ellipse_stride,
                    help="Subsampling rate for displayed metric glyphs. Smaller values draw more glyphs; larger values reduce clutter.",
                )
                ellipse_scale = st.slider(
                    "Ellipse scale",
                    0.01,
                    0.16,
                    ellipse_scale,
                    step=0.01,
                    help="Visual scale factor for ellipses. This does not change the computed distortion values.",
                )
            elif show_ellipsoids:
                max_ellipsoids = st.slider(
                    "Ellipsoids shown",
                    0,
                    45,
                    max_ellipsoids,
                    help="Maximum number of local metric ellipsoids drawn in the 3D plot.",
                )
                st.caption(
                    "Ellipsoids are drawn from evenly spaced samples. The app limits this number because "
                    "3D ellipsoids are expensive to render, so drawing one for every point can make the browser slow."
                )
                ellipse_scale = st.slider(
                    "Ellipsoid scale",
                    0.03,
                    0.30,
                    ellipse_scale,
                    step=0.01,
                    help="Visual scale factor for 3D ellipsoids. This does not change the computed distortion values.",
                )
            show_links = st.toggle(
                "Show broken neighborhood links",
                value=show_links,
                help="Show orange lines for original-space neighbors that are unusually far apart in the embedding.",
            )

if stage == "Upload raw data" and x is None:
    render_raw_upload_guide(n_components)
    st.stop()

if stage == "Evaluate my embedding" and x is None:
    render_embedding_upload_guide(n_components)
    st.stop()

try:
    with st.spinner("Computing embedding and local distortion metrics..."):
        metrics, distances, links = run_pipeline(
            x,
            meta,
            embedding_method,
            n_components,
            n_neighbors,
            affinity_radius,
            perplexity,
            outlier_factor,
            seed,
            provided_embedding,
        )
except ImportError as exc:
    st.error(str(exc))
    st.stop()

render_results(metrics, links, color_by, ellipse_stride, ellipse_scale, show_links, show_ellipsoids, max_ellipsoids, feature_export)
