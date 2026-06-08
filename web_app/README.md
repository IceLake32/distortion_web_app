# Distortions Streamlit Demo

This is the first Streamlit demo for `bbag136.pdf` and the local
`distortions` package. It focuses on the paper/package idea that nonlinear
2D embeddings can warp local geometry and fragment neighborhoods.

## Local run

```bash
pip install -r web_app/requirements.txt
streamlit run web_app/app.py
```

## Streamlit Community Cloud

Deploy this repository with `web_app/app.py` as the app entrypoint. The
dependencies are declared in `web_app/requirements.txt`, which is in the same
directory as the entrypoint file.

The app imports the local package from `../distortions`, generates toy manifold
data, computes a 2D embedding, estimates local Riemannian metric distortions,
and visualizes distortion ellipses plus broken neighborhood links.
