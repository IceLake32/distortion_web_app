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

COLOR_HELP = {
    "distortion_ratio": "Color shows anisotropic local stretching: brighter points have more elongated local metric ellipses.",
    "local_area": "Color shows the overall local metric scale: brighter points have larger local expansion/compression magnitude.",
    "manifold_position": "Color shows a reference variable from the original data, such as position along the simulated manifold or an uploaded label.",
}


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
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray | None]:
    features = df.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")
    pieces = [features]
    embedding = None
    if embedding_cols is not None:
        embedding = df.loc[:, embedding_cols].apply(pd.to_numeric, errors="coerce")
        pieces.append(embedding)

    valid = pd.concat(pieces, axis=1).replace([np.inf, -np.inf], np.nan).dropna()

    if len(valid) > max_rows:
        valid = valid.sample(n=max_rows, random_state=seed).sort_index()

    x = StandardScaler().fit_transform(valid.loc[:, feature_cols].to_numpy(dtype=float))
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
    return x, meta, y


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
    x: np.ndarray,
    meta: pd.DataFrame,
    embedding_method: str,
    n_neighbors: int,
    affinity_radius: float,
    perplexity: int,
    outlier_factor: float,
    seed: int,
    provided_embedding: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = provided_embedding if provided_embedding is not None else embed_data(x, embedding_method, n_neighbors, perplexity, seed)
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


def metric_matrix(row: pd.Series) -> np.ndarray:
    basis = np.array([[row["x0"], row["x1"]], [row["y0"], row["y1"]]], dtype=float)
    values = np.diag(np.maximum([row["s1"], row["s0"]], 1e-8))
    return basis @ values @ basis.T


def local_isometry_preview(
    df: pd.DataFrame,
    focus_ix: int,
    bandwidth: float,
    strength: float,
) -> pd.DataFrame:
    coords = df[["embedding_0", "embedding_1"]].to_numpy(dtype=float)
    focus = coords[focus_ix]
    sq_dist = np.sum((coords - focus) ** 2, axis=1)
    weights = np.exp(-sq_dist / max(2 * bandwidth**2, 1e-8))
    weights = weights / max(weights.max(), 1e-8)

    matrices = np.stack([metric_matrix(row) for _, row in df.iterrows()])
    metric_weights = weights / max(weights.sum(), 1e-8)
    h_star = np.sum(matrices * metric_weights[:, None, None], axis=0)
    eigvals, eigvecs = np.linalg.eigh(h_star)
    eigvals = np.maximum(eigvals, 1e-8)
    local_correction = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T

    corrected_raw = focus + (coords - focus) @ local_correction.T
    corrected = coords + strength * weights[:, None] * (corrected_raw - coords)
    out = pd.DataFrame(
        {
            "embedding_0": coords[:, 0],
            "embedding_1": coords[:, 1],
            "corrected_0": corrected[:, 0],
            "corrected_1": corrected[:, 1],
            "weight": weights,
            "label": df["label"].astype(str).to_numpy(),
        }
    )
    return out[out["weight"] > 0.12].copy()


def make_plot(
    df: pd.DataFrame,
    links: pd.DataFrame,
    color_by: str,
    ellipse_stride: int,
    ellipse_scale: float,
    show_links: bool,
    isometry_preview: pd.DataFrame | None = None,
    focus_ix: int | None = None,
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

    if isometry_preview is not None and len(isometry_preview) > 0:
        for row in isometry_preview.itertuples(index=False):
            fig.add_trace(
                go.Scatter(
                    x=[row.embedding_0, row.corrected_0],
                    y=[row.embedding_1, row.corrected_1],
                    mode="lines",
                    line={"color": "rgba(230, 90, 50, 0.24)", "width": 1.5},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=isometry_preview["corrected_0"],
                y=isometry_preview["corrected_1"],
                mode="markers",
                marker={
                    "size": 6 + 5 * isometry_preview["weight"],
                    "color": "rgba(230, 90, 50, 0.78)",
                    "line": {"width": 0.8, "color": "white"},
                },
                text=isometry_preview["label"],
                hovertemplate="local correction preview<br>label=%{text}<extra></extra>",
                name="local isometry preview",
            )
        )

    if focus_ix is not None:
        focus = df.iloc[int(focus_ix)]
        fig.add_trace(
            go.Scatter(
                x=[focus["embedding_0"]],
                y=[focus["embedding_1"]],
                mode="markers",
                marker={"symbol": "star", "size": 15, "color": "#e65a32", "line": {"width": 1, "color": "white"}},
                hovertemplate="isometry anchor<br>sample=%{text}<extra></extra>",
                text=[str(focus_ix)],
                name="isometry anchor",
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


st.title("Visualizing Distortions in Low-Dimensional Embeddings")
main_placeholder = st.container()

with st.sidebar:
    with st.expander("Download local version"):
        st.write(
            "For larger private datasets, download a portable version and run the app locally. "
            "No Python installation is needed after unzipping."
        )
        st.link_button("Windows zip", f"{RELEASE_BASE_URL}/DistortionsDemo_Windows.zip")
        st.link_button("macOS zip", f"{RELEASE_BASE_URL}/DistortionsDemo_macOS.zip")

    with st.expander("Citations and links"):
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

    st.subheader("Data")
    data_source = st.radio("Data source", ["Built-in examples", "Upload data"], horizontal=True)
    seed = st.number_input("Random seed", value=7, min_value=0, max_value=9999)

    if data_source == "Built-in examples":
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
        provided_embedding = None
    else:
        uploaded_file = st.file_uploader(
            "Upload data",
            type=["csv", "tsv", "txt", "xlsx", "xls"],
            help="Upload a table file. Supported formats: CSV, TSV/TXT, XLSX, and XLS.",
        )
        if uploaded_file is None:
            with main_placeholder:
                render_upload_format_guide()
            st.stop()

        try:
            raw_df = read_uploaded_table(uploaded_file.getvalue(), uploaded_file.name)
        except Exception as exc:
            st.error(f"Could not read uploaded file: {exc}")
            st.stop()
        numeric_cols = raw_df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < 2:
            st.error("The uploaded CSV needs at least two numeric feature columns.")
            st.stop()

        feature_cols = st.multiselect(
            "Feature columns",
            numeric_cols,
            default=numeric_cols,
            help="Numeric columns used as the original high-dimensional data matrix.",
        )
        if len(feature_cols) < 2:
            st.error("Select at least two feature columns.")
            st.stop()

        label_options = ["None"] + raw_df.columns.tolist()
        label_choice = st.selectbox(
            "Label / reference column",
            label_options,
            help="Optional column used for labels and reference coloring. It is not used as a feature.",
        )
        label_col = None if label_choice == "None" else label_choice

        embedding_source = st.radio(
            "Embedding source",
            ["Compute in app", "Use columns from file"],
            help="Compute PCA/Isomap/t-SNE inside the app, or select existing 2D embedding coordinates from the uploaded table.",
        )
        embedding_cols = None
        if embedding_source == "Use columns from file":
            embedding_cols = st.multiselect(
                "Embedding columns",
                numeric_cols,
                default=numeric_cols[-2:] if len(numeric_cols) >= 2 else numeric_cols,
                max_selections=2,
                help="Select exactly two numeric columns containing the uploaded 2D embedding coordinates.",
            )
            if len(embedding_cols) != 2:
                st.error("Select exactly two embedding columns.")
                st.stop()
            overlapping_cols = [col for col in embedding_cols if col in feature_cols]
            if overlapping_cols:
                feature_cols = [col for col in feature_cols if col not in embedding_cols]
                st.caption("Embedding columns are excluded from the feature matrix.")
            if len(feature_cols) < 2:
                st.error("After excluding embedding columns, select at least two feature columns.")
                st.stop()

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

        x, meta, provided_embedding = prepare_uploaded_dataset(
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

    st.subheader("Embedding")
    if provided_embedding is None:
        embedding_method = st.segmented_control(
            "Embedding",
            ["PCA", "Isomap", "t-SNE"],
            default="Isomap",
            help="Method used to compute the 2D embedding shown in the plot.",
        )
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
            "and flag broken neighbor links. For Isomap, this same value also controls the "
            "embedding graph."
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
    show_links = st.toggle(
        "Show broken neighborhood links",
        value=True,
        help="Show orange lines for original-space neighbors that are unusually far apart in the embedding.",
    )

with st.spinner("Computing embedding and local distortion metrics..."):
    metrics, distances, links = run_pipeline(
        x,
        meta,
        embedding_method,
        n_neighbors,
        affinity_radius,
        perplexity,
        outlier_factor,
        seed,
        provided_embedding,
    )

isometry_preview = None
focus_ix = None
with st.sidebar:
    st.subheader("Local isometry preview")
    show_isometry = st.toggle(
        "Show local correction preview",
        value=False,
        help=(
            "Approximate the package's inter_isometry idea: choose one anchor sample, "
            "then show how nearby points would move after a local metric correction."
        ),
    )
    if show_isometry:
        default_focus = int(metrics["distortion_ratio"].idxmax())
        focus_ix = st.number_input(
            "Anchor sample",
            min_value=0,
            max_value=len(metrics) - 1,
            value=default_focus,
            help="Sample used as the center of the local isometry correction. The default is the most distorted sample.",
        )
        correction_bw = st.slider(
            "Correction bandwidth",
            0.1,
            5.0,
            1.0,
            step=0.1,
            help="Controls how far the local correction spreads around the anchor sample in the embedding.",
        )
        correction_strength = st.slider(
            "Correction strength",
            0.0,
            1.0,
            0.7,
            step=0.05,
            help="Blends between the original embedding and the locally corrected preview. It does not recompute the embedding.",
        )
        isometry_preview = local_isometry_preview(
            metrics,
            int(focus_ix),
            correction_bw,
            correction_strength,
        )

left, right = st.columns([0.72, 0.28], gap="large")

with left:
    color_by = st.radio(
        "Color",
        ["distortion_ratio", "local_area", "manifold_position"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.caption(COLOR_HELP[color_by])
    st.plotly_chart(
        make_plot(metrics, links, color_by, ellipse_stride, ellipse_scale, show_links, isometry_preview, focus_ix),
        width="stretch",
        config={"displayModeBar": True, "scrollZoom": True},
    )

with right:
    broken_count = int(links["broken"].sum())
    st.metric("Flagged neighbor links", f"{broken_count:,}", f"{broken_count / max(len(links), 1):.1%}")
    st.metric("Median axis ratio", f"{metrics['distortion_ratio'].median():.2f}")
    st.metric("95th pct axis ratio", f"{metrics['distortion_ratio'].quantile(0.95):.2f}")

    st.subheader("How to read the visualization")
    st.write(
        "Each point is one high-dimensional sample after embedding. The ellipses summarize the local "
        "Riemannian metric estimated from the original space: elongated ellipses indicate directions "
        "that the embedding stretches unevenly. Orange links mark original-space neighbors that are "
        "unusually far apart in the embedding."
    )

    st.subheader("Most distorted samples")
    st.dataframe(
        metrics[["embedding_0", "embedding_1", "label", "distortion_ratio", "local_area"]]
        .sort_values("distortion_ratio", ascending=False)
        .head(12),
        width="stretch",
    )
