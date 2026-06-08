from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs, make_s_curve, make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
for package_root in (ROOT, ROOT / "distortions"):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from distortions.geometry import Geometry, bind_metric, local_distortions  # noqa: E402


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


def embed_data(x: np.ndarray, method: str, n_neighbors: int, perplexity: int, seed: int) -> np.ndarray:
    if method == "PCA":
        return _standardize_embedding(PCA(n_components=2, random_state=seed).fit_transform(x))
    if method == "Isomap":
        return _standardize_embedding(Isomap(n_neighbors=n_neighbors, n_components=2).fit_transform(x))
    return _standardize_embedding(
        TSNE(
            n_components=2,
            perplexity=min(perplexity, max(5, (x.shape[0] - 1) // 3)),
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(x)
    )


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
    _, h_vectors, h_values = local_distortions(y, x, geom)
    out = bind_metric(y, h_vectors, h_values)
    out["distortion_ratio"] = np.divide(
        np.maximum(out["s0"], out["s1"]),
        np.maximum(np.minimum(out["s0"], out["s1"]), 1e-8),
    )
    out["local_area"] = np.sqrt(np.maximum(out["s0"] * out["s1"], 0))
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
    dataset: str,
    n_samples: int,
    noise: float,
    embedding_method: str,
    n_neighbors: int,
    affinity_radius: float,
    perplexity: int,
    outlier_factor: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x, meta = make_dataset(dataset, n_samples, noise, seed)
    y = embed_data(x, embedding_method, n_neighbors, perplexity, seed)
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


def make_plot(
    df: pd.DataFrame,
    links: pd.DataFrame,
    color_by: str,
    ellipse_stride: int,
    ellipse_scale: float,
    show_links: bool,
) -> go.Figure:
    colors = df[color_by]
    fig = go.Figure()

    if show_links:
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

    sampled = df.iloc[::ellipse_stride]
    for _, row in sampled.iterrows():
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

    fig.add_trace(
        go.Scatter(
            x=df["embedding_0"],
            y=df["embedding_1"],
            mode="markers",
            marker={
                "size": 7,
                "color": colors,
                "colorscale": "Viridis",
                "showscale": color_by != "label",
                "line": {"width": 0.5, "color": "rgba(255,255,255,0.7)"},
            },
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


st.title("Distortions: first Streamlit demo")
st.caption("Explore how a 2D embedding stretches local geometry and breaks original-space neighborhoods.")

with st.sidebar:
    st.header("Controls")
    dataset = st.selectbox("Dataset", list(DATASET_HELP), help=DATASET_HELP["Swiss roll"])
    n_samples = st.slider("Samples", 120, 650, 360, step=30)
    noise = st.slider("Noise", 0.0, 0.5, 0.08, step=0.02)
    embedding_method = st.segmented_control("Embedding", ["PCA", "Isomap", "t-SNE"], default="Isomap")
    n_neighbors = st.slider("Neighbors", 6, 40, 14)
    affinity_radius = st.slider("Affinity radius", 0.2, 5.0, 1.6, step=0.1)
    perplexity = st.slider("t-SNE perplexity", 5, 80, 30)
    outlier_factor = st.slider("Broken-link sensitivity", 0.5, 4.0, 1.5, step=0.1)
    ellipse_stride = st.slider("Ellipse density", 1, 12, 5)
    ellipse_scale = st.slider("Ellipse scale", 0.01, 0.16, 0.05, step=0.01)
    seed = st.number_input("Random seed", value=7, min_value=0, max_value=9999)
    show_links = st.toggle("Show broken neighborhood links", value=True)

with st.spinner("Computing embedding and local distortion metrics..."):
    metrics, distances, links = run_pipeline(
        dataset,
        n_samples,
        noise,
        embedding_method,
        n_neighbors,
        affinity_radius,
        perplexity,
        outlier_factor,
        seed,
    )

left, right = st.columns([0.72, 0.28], gap="large")

with left:
    color_by = st.radio(
        "Color",
        ["distortion_ratio", "local_area", "manifold_position"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.plotly_chart(
        make_plot(metrics, links, color_by, ellipse_stride, ellipse_scale, show_links),
        use_container_width=True,
        config={"displayModeBar": True, "scrollZoom": True},
    )

with right:
    broken_count = int(links["broken"].sum())
    st.metric("Broken links", f"{broken_count:,}", f"{broken_count / max(len(links), 1):.1%}")
    st.metric("Median axis ratio", f"{metrics['distortion_ratio'].median():.2f}")
    st.metric("95th pct axis ratio", f"{metrics['distortion_ratio'].quantile(0.95):.2f}")

    st.subheader("What the professor is seeing")
    st.write(
        "Each point is one high-dimensional sample after embedding. The small ellipses summarize the local "
        "Riemannian metric estimated by the package: long, skinny ellipses mark regions where nearby "
        "directions are stretched unevenly. Orange links are original-space neighbors whose embedded "
        "distance is unusually large compared with similar true-distance pairs."
    )

    st.subheader("Top distorted samples")
    st.dataframe(
        metrics[["embedding_0", "embedding_1", "label", "distortion_ratio", "local_area"]]
        .sort_values("distortion_ratio", ascending=False)
        .head(12),
        use_container_width=True,
    )
