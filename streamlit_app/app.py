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
from app.platform import (
    batch_predict,
    drift_check,
    generate_profile_card,
    get_job_intake_template,
    list_profile_entities,
    predict_new_job_with_profiles,
    profile_data,
    train_model,
)
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


def _make_widget_key(prefix: str, field: str) -> str:
    clean = field.replace(" ", "_").replace(".", "_").replace("/", "_")
    return f"{prefix}_{clean}"


def _render_profile_card(card: dict) -> None:
    st.markdown(f"### {card['profile_type'].title()} Profile: {card['name']}")
    portfolio = card.get("portfolio", {})
    financials = card.get("financials", {})
    delivery = card.get("delivery", {})
    snapshot = card.get("snapshot", {})

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Jobs", portfolio.get("total_jobs", 0))
    m2.metric("Active Jobs", portfolio.get("active_jobs", 0))
    m3.metric("Completed Jobs", portfolio.get("completed_jobs", 0))
    m4.metric("Overdue Open Jobs", portfolio.get("overdue_open_jobs", 0))

    st.markdown("**Portfolio Health**")
    st.dataframe(
        [
            {"metric": "Completion Rate %", "value": portfolio.get("completion_rate_pct")},
            {"metric": "Cancellation Rate %", "value": portfolio.get("cancellation_rate_pct")},
            {"metric": "First Job Date", "value": snapshot.get("date_range", {}).get("first_job_date")},
            {"metric": "Last Job Date", "value": snapshot.get("date_range", {}).get("last_job_date")},
            {"metric": "Avg Cycle Time (days)", "value": delivery.get("avg_cycle_time_days")},
            {"metric": "Median Cycle Time (days)", "value": delivery.get("median_cycle_time_days")},
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Financials**")
    st.dataframe(
        [
            {"metric": "Budget Revenue Total", "value": financials.get("budget_revenue_total")},
            {"metric": "Actual Revenue Total", "value": financials.get("actual_revenue_total")},
            {"metric": "Budget Cost Total", "value": financials.get("budget_cost_total")},
            {"metric": "Actual Cost Total", "value": financials.get("actual_cost_total")},
            {"metric": "Budget Margin Total", "value": financials.get("budget_margin_total")},
            {"metric": "Actual Margin Total", "value": financials.get("actual_margin_total")},
            {"metric": "Revenue Variance %", "value": financials.get("revenue_variance_pct")},
            {"metric": "Margin Variance %", "value": financials.get("margin_variance_pct")},
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Delivery Mix**")
    st.json(
        {
            "top_statuses": delivery.get("top_statuses", {}),
            "top_budget_types": delivery.get("top_budget_types", {}),
            "top_postcodes": delivery.get("top_postcodes", {}),
            "avg_total_to_do_hours": delivery.get("avg_total_to_do_hours"),
            "avg_total_done_hours": delivery.get("avg_total_done_hours"),
        }
    )

    st.markdown("**Recent Jobs**")
    st.dataframe(card.get("recent_jobs", []), use_container_width=True)


def _page_overview() -> None:
    st.subheader("Platform Overview")
    st.markdown(
        """
        This platform supports:
        - Uploading or connecting CSV/XLSX datasets.
        - Auto-preprocessing (missing values, categorical encoding, date expansion).
        - Multi-model training and best-model selection.
        - Model registry with experiment history.
        - New job intake forms with instant predictive outcomes.
        - Detailed partner and client profile card generation.
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


def _page_new_job_intake() -> None:
    st.subheader("New Job Intake & Prediction")
    experiments = list_experiments(limit=200)
    if not experiments:
        st.info("No registered model available. Train a model first.")
        return

    experiment_id = st.selectbox(
        "Model (experiment id)",
        options=[item["id"] for item in experiments],
        key="intake_experiment_id",
    )
    max_fields = st.slider("Form field count", min_value=8, max_value=40, value=18, step=2)

    template_key = f"intake_template_{experiment_id}_{max_fields}"
    if st.button("Load Intake Form", type="primary"):
        try:
            st.session_state[template_key] = get_job_intake_template(
                experiment_id=experiment_id,
                max_fields=max_fields,
            )
        except Exception as exc:
            st.error(str(exc))
            return

    template = st.session_state.get(template_key)
    if not template:
        st.info("Load intake form to enter a new job.")
        return

    st.caption(
        f"Target being predicted: `{template['target_column']}` "
        f"({template['task_type']})"
    )
    profile_dataset_path = st.text_input(
        "Profile source dataset path",
        value=template["dataset_path"],
        key="intake_profile_source_path",
    )

    draft_record: dict[str, str] = {}
    with st.form("new_job_prediction_form"):
        for field_meta in template["recommended_fields"]:
            field_name = field_meta["field"]
            field_type = field_meta["type"]
            default = field_meta["default"]
            required = field_meta["required"]
            label = f"{field_name}{' *' if required else ''}"
            key = _make_widget_key("intake", field_name)

            if field_type == "category":
                options = [""] + [str(v) for v in field_meta.get("suggested_values", [])]
                default_option = str(default) if default else ""
                index = options.index(default_option) if default_option in options else 0
                draft_record[field_name] = st.selectbox(label, options=options, index=index, key=key)
            else:
                placeholder = ""
                if field_type == "number" and default not in (None, ""):
                    placeholder = str(round(float(default), 4))
                elif field_type == "date":
                    placeholder = "dd/mm/yyyy"
                draft_record[field_name] = st.text_input(label, value="", placeholder=placeholder, key=key)

        submitted = st.form_submit_button("Predict New Job", type="primary")

    if not submitted:
        return

    clean_record: dict[str, object] = {}
    for field_meta in template["recommended_fields"]:
        name = field_meta["field"]
        value = draft_record.get(name, "")
        field_type = field_meta["type"]

        if isinstance(value, str):
            value = value.strip()
        if value in {"", None}:
            continue

        if field_type == "number":
            try:
                clean_record[name] = float(value)
            except Exception:
                clean_record[name] = value
        else:
            clean_record[name] = value

    if not clean_record:
        st.warning("Enter at least one field value for prediction.")
        return

    with st.spinner("Predicting and generating profile cards..."):
        try:
            result = predict_new_job_with_profiles(
                experiment_id=experiment_id,
                job_record=clean_record,
                dataset_path=profile_dataset_path,
            )
        except Exception as exc:
            st.error(str(exc))
            return

    prediction = result["prediction"]
    first_prediction = prediction["predictions"][0] if prediction["predictions"] else None
    confidence = prediction.get("confidence", [None])[0]
    c1, c2 = st.columns(2)
    c1.metric("Predicted Outcome", str(first_prediction))
    c2.metric(
        "Confidence",
        f"{confidence:.2%}" if isinstance(confidence, float) else "N/A",
    )
    st.markdown("**Submitted Job Record**")
    st.json(clean_record)

    profiles = result.get("profiles", {})
    if "client" in profiles:
        _render_profile_card(profiles["client"])
    elif "client_error" in profiles:
        st.warning(f"Client profile not generated: {profiles['client_error']}")

    if "partner" in profiles:
        _render_profile_card(profiles["partner"])
    elif "partner_error" in profiles:
        st.warning(f"Partner profile not generated: {profiles['partner_error']}")


def _page_profiles() -> None:
    st.subheader("Partner & Client Profile Cards")
    dataset_path = st.text_input(
        "Dataset path",
        value=str(DEFAULT_DATASET_PATH),
        key="profiles_dataset_path",
    )

    tab_client, tab_partner = st.tabs(["Client Profiles", "Partner Profiles"])

    with tab_client:
        if st.button("Load Client List"):
            try:
                st.session_state["client_entities"] = list_profile_entities(
                    dataset_path=dataset_path,
                    profile_type="client",
                    limit=400,
                )
            except Exception as exc:
                st.error(str(exc))

        client_entities = st.session_state.get("client_entities", [])
        client_name = st.selectbox(
            "Client",
            options=client_entities if client_entities else [""],
            key="client_profile_select",
        )
        client_name_override = st.text_input("Or type client name manually", value="", key="client_profile_manual")
        resolved_client = client_name_override.strip() or client_name

        if st.button("Generate Client Profile Card", type="primary"):
            if not resolved_client:
                st.warning("Choose or type a client name.")
            else:
                try:
                    card = generate_profile_card(
                        dataset_path=dataset_path,
                        profile_type="client",
                        name=resolved_client,
                    )
                    _render_profile_card(card)
                except Exception as exc:
                    st.error(str(exc))

    with tab_partner:
        if st.button("Load Partner List"):
            try:
                st.session_state["partner_entities"] = list_profile_entities(
                    dataset_path=dataset_path,
                    profile_type="partner",
                    limit=400,
                )
            except Exception as exc:
                st.error(str(exc))

        partner_entities = st.session_state.get("partner_entities", [])
        partner_name = st.selectbox(
            "Partner",
            options=partner_entities if partner_entities else [""],
            key="partner_profile_select",
        )
        partner_name_override = st.text_input("Or type partner name manually", value="", key="partner_profile_manual")
        resolved_partner = partner_name_override.strip() or partner_name

        if st.button("Generate Partner Profile Card", type="primary"):
            if not resolved_partner:
                st.warning("Choose or type a partner name.")
            else:
                try:
                    card = generate_profile_card(
                        dataset_path=dataset_path,
                        profile_type="partner",
                        name=resolved_partner,
                    )
                    _render_profile_card(card)
                except Exception as exc:
                    st.error(str(exc))


def main() -> None:
    _header()
    page = st.sidebar.radio(
        "Navigate",
        options=[
            "Overview",
            "Dataset Profiling",
            "Training Studio",
            "Model Registry",
            "New Job Intake",
            "Profile Cards",
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
    elif page == "New Job Intake":
        _page_new_job_intake()
    elif page == "Profile Cards":
        _page_profiles()
    elif page == "Batch Prediction":
        _page_batch_prediction()
    else:
        _page_drift()


if __name__ == "__main__":
    main()
