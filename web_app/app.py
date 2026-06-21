from __future__ import annotations

import io
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
    "manifold_position": "Color shows a reference variable from the original data, such as position along the simulated manifold or an uploaded label.",
}

EXAMPLE_DIR = Path(__file__).resolve().parent / "example_data"
RAW_DATA_EXAMPLE_PATH = EXAMPLE_DIR / "raw_swiss_roll_example.csv"
EMBEDDING_DATA_EXAMPLE_PATH = EXAMPLE_DIR / "existing_embedding_swiss_roll_example.csv"


def render_upload_format_guide() -> None:
    st.subheader("Upload data format")
    st.write(
        "Upload a table where each row is one sample. Numeric columns can be used as the "
        "original high-dimensional features. You can either let the app compute a 2D "
        "embedding, or provide two existing embedding coordinate columns."
    )

    st.markdown("**Supported files:** CSV, TSV/TXT, XLSX, and XLS.")

    compute_tab, embedding_tab = st.tabs(["App computes embedding", "Use uploaded embedding"])
    with compute_tab:
        st.write("Use this format when the app should compute PCA, Isomap, or t-SNE.")
        st.code(
            """feature_1,feature_2,feature_3,label
0.1,1.2,0.4,A
0.3,1.0,0.5,A
2.4,0.2,1.1,B""",
            language="csv",
        )
        st.markdown(
            """
            Select the numeric feature columns as `Feature columns`. The optional `label`
            column can be selected as `Label / reference column`.
            """
        )

    with embedding_tab:
        st.write("Use this format when you already have UMAP, t-SNE, PCA, or other 2D coordinates.")
        st.code(
            """feature_1,feature_2,feature_3,umap_1,umap_2,label
0.1,1.2,0.4,-2.1,0.5,A
0.3,1.0,0.5,-1.9,0.7,A
2.4,0.2,1.1,1.4,-0.3,B""",
            language="csv",
        )
        st.markdown(
            """
            Select the original numeric variables as `Feature columns`, then choose the two
            embedding coordinate columns as `Embedding columns`. The embedding columns are
            excluded from the feature matrix automatically.
            """
        )


def results_csv(metrics: pd.DataFrame, features: pd.DataFrame | None = None) -> bytes:
    out = metrics.reset_index(names="sample_index")
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
        meta = pd.DataFrame({"label": "roll", "manifold_position": color})
        return StandardScaler().fit_transform(x), meta

    if name == "S-curve":
        x, color = make_s_curve(n_samples=n_samples, noise=noise, random_state=seed)
        meta = pd.DataFrame({"label": "curve", "manifold_position": color})
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
    meta = pd.DataFrame({"label": np.where(labels == 0, "cluster A", "cluster B"), "manifold_position": labels})
    return StandardScaler().fit_transform(x), meta


def prepare_uploaded_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str | None,
    embedding_cols: list[str] | None,
    max_rows: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray | None, pd.DataFrame]:
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
        labels = df.loc[valid.index, label_col].astype(str).fillna("missing").reset_index(drop=True)
        manifold_position = pd.factorize(labels)[0]
    else:
        labels = pd.Series(["uploaded"] * len(valid))
        manifold_position = np.arange(len(valid))

    meta = pd.DataFrame({
        "label": labels,
        "manifold_position": manifold_position,
    })
    return x, meta, y, feature_export


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
) -> go.Figure:
    colors = df[color_by]
    fig = go.Figure()
    is_3d = "embedding_2" in df.columns

    if show_links:
        broken = links[links["broken"]].head(180)
        for edge in broken.itertuples(index=False):
            a = df.iloc[int(edge.center)]
            b = df.iloc[int(edge.neighbor)]
            if is_3d:
                fig.add_trace(
                    go.Scatter3d(
                        x=[a.embedding_0, b.embedding_0],
                        y=[a.embedding_1, b.embedding_1],
                        z=[a.embedding_2, b.embedding_2],
                        mode="lines",
                        line={"color": "rgba(228, 87, 46, 0.20)", "width": 2},
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
            else:
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

    if not is_3d:
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

    if is_3d and show_ellipsoids:
        for _, row in df.iloc[::ellipse_stride].head(45).iterrows():
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

    marker = {
        "size": 7,
        "color": colors,
        "colorscale": "Viridis",
        "showscale": color_by != "label",
        "line": {"width": 0.5, "color": "rgba(255,255,255,0.7)"},
    }
    if is_3d:
        fig.add_trace(
            go.Scatter3d(
                x=df["embedding_0"],
                y=df["embedding_1"],
                z=df["embedding_2"],
                mode="markers",
                marker=marker,
                text=df["label"],
                customdata=np.stack([df["distortion_ratio"], df["local_area"]], axis=1),
                hovertemplate=(
                    "x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<br>"
                    "label=%{text}<br>"
                    "axis ratio=%{customdata[0]:.2f}<br>"
                    "local scale=%{customdata[1]:.2f}<extra></extra>"
                ),
                name="samples",
            )
        )
        fig.update_layout(
            height=690,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            paper_bgcolor="white",
            scene={
                "xaxis": {"title": ""},
                "yaxis": {"title": ""},
                "zaxis": {"title": ""},
                "aspectmode": "data",
            },
            legend={"orientation": "h", "y": 1.02},
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df["embedding_0"],
                y=df["embedding_1"],
                mode="markers",
                marker=marker,
                text=df["label"],
                customdata=np.stack([df["distortion_ratio"], df["local_area"]], axis=1),
                hovertemplate=(
                    "x=%{x:.2f}<br>y=%{y:.2f}<br>"
                    "label=%{text}<br>"
                    "axis ratio=%{customdata[0]:.2f}<br>"
                    "local area=%{customdata[1]:.2f}<extra></extra>"
                ),
                name="samples",
            )
        )
        fig.update_layout(
            height=690,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            paper_bgcolor="white",
            plot_bgcolor="#fbfaf7",
            xaxis={"title": "", "showgrid": False, "zeroline": False},
            yaxis={"title": "", "showgrid": False, "zeroline": False, "scaleanchor": "x", "scaleratio": 1},
            legend={"orientation": "h", "y": 1.02},
        )
    return fig


def render_downloads_and_citations() -> None:
    st.subheader("Run locally")
    st.write(
        "For larger private datasets, download a portable version and run the app locally. "
        "No Python package installation is needed after unzipping."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.link_button("Download Windows zip", f"{RELEASE_BASE_URL}/DistortionsDemo_Windows.zip")
    with c2:
        st.link_button("Download macOS zip", f"{RELEASE_BASE_URL}/DistortionsDemo_macOS.zip")

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
    feature_cols = st.multiselect(
        "Original feature columns",
        numeric_cols,
        default=default_features,
        help="Numeric columns used as the original high-dimensional data matrix.",
    )
    st.caption("These columns define the original data space used for embedding and distortion estimation.")
    feature_cols = [col for col in feature_cols if col not in (embedding_cols or [])]
    if len(feature_cols) < 2:
        st.error("Select at least two feature columns. Embedding coordinate columns should not be used as features.")
        st.stop()

    label_options = ["None"] + raw_df.columns.tolist()
    label_choice = st.selectbox(
        "Label / reference column",
        label_options,
        index=label_options.index(default_label_column(raw_df)) if default_label_column(raw_df) in label_options else 0,
        help="Optional column used for labels and reference coloring. It is not used as a feature.",
    )
    st.caption("This column is only used for labels, tooltips, and reference coloring; it is not used in geometry calculations.")
    label_col = None if label_choice == "None" else label_choice

    row_cap = min(len(raw_df), 1500)
    default_rows = min(len(raw_df), 400)
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
        max_rows = row_cap

    return feature_cols, label_col, embedding_cols, max_rows


def render_results(
    metrics: pd.DataFrame,
    links: pd.DataFrame,
    color_by: str,
    ellipse_stride: int,
    ellipse_scale: float,
    show_links: bool,
    show_ellipsoids: bool = False,
    feature_export: pd.DataFrame | None = None,
) -> None:
    left, right = st.columns([0.72, 0.28], gap="large")

    with left:
        st.plotly_chart(
            make_plot(metrics, links, color_by, ellipse_stride, ellipse_scale, show_links, show_ellipsoids),
            width="stretch",
            config={"displayModeBar": True, "scrollZoom": True},
        )

    with right:
        is_3d = "embedding_2" in metrics.columns
        broken_count = int(links["broken"].sum())
        st.metric("Flagged neighbor links", f"{broken_count:,}", f"{broken_count / max(len(links), 1):.1%}")
        st.caption("Original-space nearest-neighbor pairs whose embedded distance is unusually large.")
        st.metric("Median axis ratio", f"{metrics['distortion_ratio'].median():.2f}")
        st.caption("Typical local anisotropy. Larger values mean the local metric ellipse is more elongated.")
        st.metric("95th pct axis ratio", f"{metrics['distortion_ratio'].quantile(0.95):.2f}")
        st.caption("Upper-tail local anisotropy. This highlights severe distortion among the most stretched samples.")

        st.subheader("How to read the visualization")
        if is_3d:
            st.write(
                "Each point is one high-dimensional sample after embedding into 3D. Color summarizes local "
                "metric distortion estimated from the original space. Orange links mark original-space "
                "neighbors that are unusually far apart in the 3D embedding."
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
            help="Download feature columns, plotted 2D embedding coordinates, and distortion metrics.",
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
    seed = st.number_input("Random seed", value=7, min_value=0, max_value=9999)
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
    if stage == "Demo":
        dataset = st.selectbox(
            "Dataset",
            list(DATASET_HELP),
            help="Synthetic example used for the talk demo. Swiss roll and S-curve have known manifold structure; Two clusters shows cluster-level distortion.",
        )
        n_samples = st.slider(
            "Samples",
            120,
            650,
            240,
            step=30,
            help="Number of simulated samples. Larger values make the plot denser and the computation slower.",
        )
        noise = st.slider(
            "Noise",
            0.0,
            0.5,
            0.08,
            step=0.02,
            help="Amount of noise added to the synthetic data before embedding.",
        )
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
        else:
            raw_df = read_upload_or_stop(uploaded_file)
            with st.expander("Preview uploaded table", expanded=False):
                st.dataframe(raw_df.head(8), width="stretch")
            feature_cols, label_col, embedding_cols, max_rows = select_uploaded_columns(
                raw_df,
                require_embedding=(stage == "Evaluate my embedding"),
                n_components=n_components,
            )
            x, meta, provided_embedding, feature_export = prepare_uploaded_dataset(
                raw_df,
                feature_cols,
                label_col,
                embedding_cols,
                max_rows,
                seed,
            )
            if x.shape[0] < 10:
                st.error("After dropping missing values, at least 10 rows are needed for this demo.")
                st.stop()
            st.caption(f"Using {x.shape[0]} rows and {x.shape[1]} numeric features.")

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

        if embedding_method == "t-SNE":
            max_perplexity = max(5, min(80, (x.shape[0] - 1) // 3))
            perplexity = st.slider(
                "t-SNE perplexity",
                5,
                max_perplexity,
                min(30, max_perplexity),
                help="t-SNE neighborhood scale. Higher values make t-SNE consider broader neighborhoods.",
            )
        else:
            perplexity = 30

        st.subheader("Distortion estimation")
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
            1.6,
            step=0.1,
            help="Radius for the Gaussian affinity kernel used when constructing the original-space graph Laplacian.",
        )
        outlier_factor = st.slider(
            "Broken-link sensitivity",
            0.5,
            4.0,
            1.5,
            step=0.1,
            help="Controls the boxplot-style outlier threshold for flagged neighbor links. Smaller values flag more links.",
        )

        st.subheader("Visualization")
        color_by = st.radio(
            "Color",
            ["distortion_ratio", "local_area", "manifold_position"],
            horizontal=False,
            help="Choose what point color represents in the embedding plot.",
        )
        st.caption(COLOR_HELP[color_by])
        if n_components == 2:
            show_ellipsoids = False
            ellipse_stride = st.slider(
                "Ellipse density",
                1,
                12,
                5,
                help="Subsampling rate for displayed metric glyphs. Smaller values draw more glyphs; larger values reduce clutter.",
            )
            ellipse_scale = st.slider(
                "Ellipse scale",
                0.01,
                0.16,
                0.05,
                step=0.01,
                help="Visual scale factor for ellipses. This does not change the computed distortion values.",
            )
        else:
            show_ellipsoids = st.toggle(
                "Show 3D ellipsoid glyphs",
                value=False,
                help="Draw a sparse set of translucent ellipsoids summarizing the local 3D metric. This can make the plot slower.",
            )
            st.caption(
                "3D mode uses point color and broken-neighborhood links to show distortion. "
                "Optional ellipsoids show the local 3D metric, but can add visual clutter."
            )
            ellipse_stride = st.slider(
                "Ellipsoid density",
                4,
                30,
                12,
                help="Subsampling rate for displayed 3D ellipsoids. Larger values draw fewer ellipsoids.",
            )
            ellipse_scale = st.slider(
                "Ellipsoid scale",
                0.01,
                0.20,
                0.04,
                step=0.01,
                help="Visual scale factor for 3D ellipsoids. This does not change the computed distortion values.",
            )
        show_links = st.toggle(
            "Show broken neighborhood links",
            value=True,
            help="Show orange lines for original-space neighbors that are unusually far apart in the embedding.",
        )

if stage == "Upload raw data" and x is None:
    st.subheader("Upload raw data")
    st.write(
        "Use this workflow when you have original data but do not already have an embedding. "
        f"The app will compute a {n_components}D embedding from your selected original feature columns, then evaluate its distortions."
    )
    st.markdown(
        """
        **Required format**
        - Rows are samples.
        - Numeric feature columns define the original data space.
        - An optional label/reference column is used only for coloring and tooltips.
        - Embedding coordinates are not required in this workflow.
        """
    )
    st.download_button(
        "Download synthetic raw data example",
        data=example_csv_bytes(str(RAW_DATA_EXAMPLE_PATH)),
        file_name="raw_swiss_roll_example.csv",
        mime="text/csv",
        help="Download a 240-row synthetic Swiss roll table showing the expected raw data format.",
    )
    st.dataframe(example_preview(str(RAW_DATA_EXAMPLE_PATH)), width="stretch")
    st.stop()

if stage == "Evaluate my embedding" and x is None:
    st.subheader("Evaluate my embedding")
    st.write(
        f"Use this workflow when you already computed a {n_components}D embedding elsewhere. "
        "The app will not recompute the embedding; it will evaluate the one you upload."
    )
    st.markdown(
        """
        **Required format**
        - Rows are samples.
        - Original feature columns define the high-dimensional data space.
        - Embedding columns define the layout to evaluate. Use the sidebar to choose 2D or 3D.
        - An optional label/reference column is used only for coloring and tooltips.
        """
    )
    st.download_button(
        "Download synthetic embedding example",
        data=example_csv_bytes(str(EMBEDDING_DATA_EXAMPLE_PATH)),
        file_name="existing_embedding_swiss_roll_example.csv",
        mime="text/csv",
        help="Download a 240-row synthetic Swiss roll table with raw features plus existing Isomap coordinates.",
    )
    st.dataframe(example_preview(str(EMBEDDING_DATA_EXAMPLE_PATH)), width="stretch")
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

render_results(metrics, links, color_by, ellipse_stride, ellipse_scale, show_links, show_ellipsoids, feature_export)
