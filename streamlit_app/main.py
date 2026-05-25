from __future__ import annotations

import sys
from pathlib import Path

# Ensure `app.*` imports resolve to the project package, not this Streamlit script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "app" in sys.modules:
    app_module = sys.modules["app"]
    app_file = getattr(app_module, "__file__", "") or ""
    if app_file.endswith("streamlit_app/app.py"):
        del sys.modules["app"]

import streamlit as st

from app.config import DEFAULT_DATASET_PATH, UPLOADS_DIR
from app.platform import batch_predict, drift_check, profile_data, train_model
from app.registry import get_experiment, list_experiments


st.set_page_config(
    page_title="PVF ML Platform",
    page_icon="📈",
    layout="wide",
)


def _header() -> None:
    st.title("PVF Machine Learning Platform")
    st.caption(
        "Dataset profiling, automated model training, model registry, batch scoring, and drift monitoring."
    )


def _page_overview() -> None:
    st.subheader("Platform Overview")
    st.markdown(
        """
        This platform supports:
        - Uploading or connecting CSV/XLSX datasets.
        - Auto-preprocessing (missing values, categorical encoding, date expansion).
        - Multi-model training and best-model selection.
        - Model registry with experiment history.
        - Real-time and batch predictions.
        - Data drift monitoring against training baseline.
        """
    )

    default_path = str(DEFAULT_DATASET_PATH)
    st.info(f"Default dataset path detected: `{default_path}`")


def _page_profile() -> None:
    st.subheader("Dataset Profiling")
    dataset_path = st.text_input(
        "Dataset path",
        value=str(DEFAULT_DATASET_PATH),
        key="profile_dataset_path",
    )

    if st.button("Profile Dataset", type="primary"):
        try:
            profile = profile_data(dataset_path)
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Rows", f"{profile['rows']:,}")
            col_b.metric("Columns", f"{profile['columns']:,}")
            col_c.metric("Target Candidates", f"{len(profile['target_candidates']):,}")

            st.markdown("**Dtype Breakdown**")
            st.json(profile["dtype_breakdown"])

            st.markdown("**Top Missing Columns (%)**")
            st.dataframe(
                {
                    "column": list(profile["top_missing_pct"].keys()),
                    "missing_pct": list(profile["top_missing_pct"].values()),
                },
                use_container_width=True,
            )

            st.markdown("**Candidate Targets**")
            st.dataframe(profile["target_candidates"], use_container_width=True)
        except Exception as exc:
            st.error(str(exc))


def _page_training() -> None:
    st.subheader("Training Studio")
    dataset_path = st.text_input(
        "Dataset path",
        value=str(DEFAULT_DATASET_PATH),
        key="train_dataset_path",
    )

    if st.button("Load Target Candidates"):
        try:
            st.session_state["training_profile"] = profile_data(dataset_path)
        except Exception as exc:
            st.error(str(exc))

    profile = st.session_state.get("training_profile")
    if not profile:
        st.info("Load target candidates to start training.")
        return

    candidates = profile["target_candidates"]
    if not candidates:
        st.warning("No suitable target candidates detected in this dataset.")
        return

    target_column = st.selectbox("Target column", options=candidates)
    task_option = st.selectbox("Task type", options=["auto", "classification", "regression"])
    test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    random_state = st.number_input("Random state", min_value=1, max_value=9999, value=42)

    if st.button("Train & Register Best Model", type="primary"):
        with st.spinner("Training models and registering best model..."):
            try:
                result = train_model(
                    dataset_path=dataset_path,
                    target_column=target_column,
                    task_type=None if task_option == "auto" else task_option,
                    test_size=test_size,
                    random_state=random_state,
                )
            except Exception as exc:
                st.error(str(exc))
                return

        experiment = result["experiment"]
        st.success(
            "Training complete. "
            f"Experiment `{experiment['id']}` registered with model `{result['best_model_name']}`."
        )

        st.markdown("**Best Metrics**")
        st.json(result["best_metrics"])

        st.markdown("**Model Leaderboard**")
        st.dataframe(result["all_results"], use_container_width=True)

        if result["feature_importance"]:
            st.markdown("**Top Feature Importance**")
            st.dataframe(result["feature_importance"], use_container_width=True)


def _page_registry() -> None:
    st.subheader("Model Registry")
    experiments = list_experiments(limit=200)
    if not experiments:
        st.info("No experiments yet. Train a model first.")
        return

    st.dataframe(experiments, use_container_width=True)
    experiment_id = st.selectbox(
        "Open experiment details",
        options=[item["id"] for item in experiments],
    )
    details = get_experiment(experiment_id)
    st.json(details)


def _page_batch_prediction() -> None:
    st.subheader("Batch Prediction")
    experiments = list_experiments(limit=200)
    if not experiments:
        st.info("No registered model available.")
        return

    experiment_id = st.selectbox(
        "Model (experiment id)",
        options=[item["id"] for item in experiments],
        key="batch_experiment_id",
    )

    uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv", "tsv", "txt", "xlsx", "xls"])
    if uploaded is None:
        return

    if st.button("Run Batch Scoring", type="primary"):
        temp_path = UPLOADS_DIR / uploaded.name
        temp_path.write_bytes(uploaded.getbuffer())

        with st.spinner("Generating predictions..."):
            try:
                output_df = batch_predict(experiment_id=experiment_id, dataset_path=temp_path)
            except Exception as exc:
                st.error(str(exc))
                return

        st.success(f"Scored {len(output_df):,} records.")
        st.dataframe(output_df.head(1000), use_container_width=True)
        csv_bytes = output_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Predictions CSV",
            data=csv_bytes,
            file_name=f"{experiment_id}_predictions.csv",
            mime="text/csv",
        )


def _page_drift() -> None:
    st.subheader("Drift Monitoring")
    experiments = list_experiments(limit=200)
    if not experiments:
        st.info("No registered model available.")
        return

    experiment_id = st.selectbox(
        "Model (experiment id)",
        options=[item["id"] for item in experiments],
        key="drift_experiment_id",
    )
    dataset_path = st.text_input(
        "Current dataset path",
        value=str(DEFAULT_DATASET_PATH),
        key="drift_dataset_path",
    )

    if st.button("Run Drift Check", type="primary"):
        with st.spinner("Evaluating drift..."):
            try:
                report = drift_check(experiment_id, dataset_path)
            except Exception as exc:
                st.error(str(exc))
                return

        score = report["overall_drift_score"]
        st.metric("Overall Drift Score", f"{score:.3f}")
        if report["is_alert"]:
            st.error("Drift alert triggered.")
        else:
            st.success("No significant drift alert.")

        st.markdown("**Top Drifted Features**")
        st.dataframe(report["top_drifted_features"], use_container_width=True)


def main() -> None:
    _header()
    page = st.sidebar.radio(
        "Navigate",
        options=[
            "Overview",
            "Dataset Profiling",
            "Training Studio",
            "Model Registry",
            "Batch Prediction",
            "Drift Monitoring",
        ],
    )

    if page == "Overview":
        _page_overview()
    elif page == "Dataset Profiling":
        _page_profile()
    elif page == "Training Studio":
        _page_training()
    elif page == "Model Registry":
        _page_registry()
    elif page == "Batch Prediction":
        _page_batch_prediction()
    else:
        _page_drift()


if __name__ == "__main__":
    main()

