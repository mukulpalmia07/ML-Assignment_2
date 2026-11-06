"""Streamlit front end (`streamlit run app.py`): batch CSV evaluation and single-patient prediction."""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from app.config.config import settings
from app.services.data_processor import DataProcessor
from app.services.evaluation_service import Evaluator
from app.services.model_service import ModelService

PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]  # dark-surface categorical slots, fixed order per model
STATUS = {"Benign": "#22c55e", "Malignant": "#ef4444"}
MUTED = "#64748b"
CSS = Path(__file__).parent / "assets" / "style.css"
GROUPS = {"Mean": "mean {}", "Standard error": "{} error", "Worst": "worst {}"}
MODEL_INFO = {
    "Logistic Regression": "Linear model on scaled features. Fast, interpretable baseline.",
    "Decision Tree": "Rule-based splits (depth ≤ 5). Easy to explain, less accurate.",
    "KNN": "Votes of the 7 nearest training samples in scaled feature space.",
    "Naive Bayes": "Gaussian likelihoods, assumes features are independent.",
    "Support Vector Machine": "RBF-kernel max-margin classifier with calibrated probabilities.",
}


@st.cache_resource
def get_service() -> ModelService:
    return ModelService.load()


@st.cache_data
def reference_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Bundled test set, saved benchmark metrics and feature medians (form defaults)."""
    test = pd.read_csv(settings.test_data_path)
    return test, pd.read_csv(settings.metrics_path, index_col=0), pd.read_csv(settings.data_path).drop(columns=settings.target).median()


class Dashboard:
    def __init__(self):
        self.service = get_service()
        self.test_df, self.benchmark, self.medians = reference_data()
        self.colors = alt.Scale(domain=list(self.service.models), range=PALETTE)
        self.model = None

    # ---------- layout ----------
    def run(self) -> None:
        st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🩺", layout="wide")
        st.markdown(f"<style>{CSS.read_text()}</style>", unsafe_allow_html=True)
        self.sidebar()
        st.markdown(
            '<div class="hero"><h1>🩺 Breast Cancer Classifier</h1>'
            "<p>Predict whether a breast tumour is benign or malignant from 30 cell-nucleus measurements.</p>"
            '<span class="chip">UCI WDBC dataset</span><span class="chip">5 ML models</span>'
            '<span class="chip">Pre-trained · no retraining</span><span class="chip">Batch & single prediction</span></div>',
            unsafe_allow_html=True,
        )
        best = self.benchmark["MCC"].idxmax()
        for col, (label, value, help_) in zip(st.columns(4), [
            ("Training samples", "569", "UCI Breast Cancer Wisconsin (Diagnostic)"),
            ("Features", "30", "10 nucleus traits × mean / standard error / worst"),
            ("Models", str(len(self.service.models)), "Pre-trained. Uploaded data is never used for training."),
            ("Best on test set", "".join(w[0] for w in best.split()) if " " in best else best, f"{best} · MCC {self.benchmark.loc[best, 'MCC']:.3f} · Accuracy {self.benchmark.loc[best, 'Accuracy']:.1%}"),
        ]):
            col.metric(label, value, help=help_)
        batch, single = st.tabs(["📂 Batch evaluation (CSV)", "🧪 Single patient prediction"])
        with batch:
            self.batch_tab()
        with single:
            self.single_tab()

    def sidebar(self) -> None:
        with st.sidebar:
            st.header("⚙️ Model")
            self.model = st.selectbox("Select model", list(self.service.models), index=4)
            st.caption(MODEL_INFO[self.model])
            st.divider()
            st.subheader("How to use")
            st.markdown("".join(f'<div class="step"><span>{i}</span><div>{t}</div></div>' for i, t in enumerate([
                "<b>Batch:</b> use the sample test data or upload your own CSV.",
                "<b>Single patient:</b> enter measurements and click <b>Predict</b>.",
                "Switch models here. Every view updates instantly.",
            ], 1)), unsafe_allow_html=True)
            st.divider()
            st.download_button("⬇️ Sample test_data.csv", settings.test_data_path.read_bytes(), "test_data.csv", "text/csv", width="stretch")

    # ---------- batch ----------
    def batch_tab(self) -> None:
        source = st.segmented_control("Data source", ["Sample test data (114 rows)", "Upload CSV"], default="Sample test data (114 rows)")
        df = self.test_df
        if source == "Upload CSV":
            upload = st.file_uploader("Upload a CSV with the 30 feature columns. Add a `diagnosis` column (0/1) to get evaluation metrics.", type="csv")
            with st.expander("Required columns"):
                st.code(", ".join(self.service.features + [settings.target]), language=None)
            if upload is None:
                st.info("Upload a CSV to get started, or download the sample file from the sidebar.", icon="📄")
                return
            try:
                df = pd.read_csv(upload)
            except ValueError as e:
                st.error(f"Could not read the CSV: {e}", icon="🚫")
                return
        try:
            X, y = DataProcessor.split_xy(df, self.service.features)
        except ValueError as e:
            st.error(str(e), icon="🚫")
            return

        probs = self.service.predict_proba(X)  # one pass per model, reused by every view below
        n_mal = int(Evaluator.predict(probs[self.model]).sum())
        a, b, c = st.columns(3)
        a.metric("Rows scored", len(df))
        b.metric("Predicted malignant", f"{n_mal}  ({n_mal / len(df):.0%})", help=f"Using {self.model}")
        if y is None:
            c.metric("Ground truth", "Not provided", help="Add a `diagnosis` column to enable metrics")
        else:
            c.metric(f"Accuracy · {self.model}", f"{Evaluator.metrics(y, probs[self.model])['Accuracy']:.1%}")

        if y is None:
            st.info("This file has no `diagnosis` column, so only predictions are shown. Add it to see metrics.", icon="ℹ️")
            self.predictions(df, probs)
            return
        metrics = Evaluator.compare(y, probs)
        compare, detail, preds = st.tabs(["📊 Model comparison", f"🔍 {self.model} details", "📋 Predictions"])
        with compare:
            self.comparison(y, probs, metrics)
        with detail:
            self.details(y, probs, metrics)
        with preds:
            self.predictions(df, probs)

    def comparison(self, y: pd.Series, probs: pd.DataFrame, metrics: pd.DataFrame) -> None:
        st.dataframe(metrics.style.format("{:.4f}", na_rep="n/a").highlight_max(axis=0, props="font-weight:700;color:#a78bfa"), width="stretch")
        st.caption("**Bold violet** marks the best value in each column.")
        left, right = st.columns(2)
        with left:
            metric = st.segmented_control("Metric", list(metrics.columns), default="MCC", key="cmp_metric") or "MCC"
            data = metrics[metric].rename("value").rename_axis("Model").reset_index()
            base = alt.Chart(data).encode(
                x=alt.X("value:Q", title=metric, scale=alt.Scale(domain=[min(0, data.value.min()), 1])),
                y=alt.Y("Model:N", sort=list(data.sort_values("value", ascending=False).Model), title=None, axis=alt.Axis(labelLimit=200)),
                tooltip=["Model", alt.Tooltip("value:Q", format=".4f", title=metric)],
            )
            bars = base.mark_bar(cornerRadiusEnd=4).encode(color=alt.Color("Model:N", scale=self.colors, legend=None))
            st.altair_chart(bars + base.mark_text(align="left", dx=4, color="#e7eaf3").encode(text=alt.Text("value:Q", format=".3f")).properties(height=260), width="stretch")
        with right:
            st.markdown("**ROC curves**")
            roc = alt.Chart(Evaluator.roc(y, probs)).mark_line(strokeWidth=2, interpolate="step-after").encode(
                x=alt.X("FPR:Q", title="False positive rate"), y=alt.Y("TPR:Q", title="True positive rate"),
                color=alt.Color("Model:N", scale=self.colors, legend=alt.Legend(orient="bottom", columns=3, title=None, labelLimit=200)),
                tooltip=["Model", alt.Tooltip("FPR:Q", format=".3f"), alt.Tooltip("TPR:Q", format=".3f")],
            )
            diag = alt.Chart(pd.DataFrame({"FPR": [0, 1], "TPR": [0, 1]})).mark_line(strokeDash=[4, 4], color=MUTED).encode(x="FPR:Q", y="TPR:Q")
            st.altair_chart((diag + roc).properties(height=320), width="stretch")

    def details(self, y: pd.Series, probs: pd.DataFrame, metrics: pd.DataFrame) -> None:
        for col, (k, v) in zip(st.columns(len(metrics.columns)), metrics.loc[self.model].items()):
            gap = v - metrics[k].max()
            col.metric(k, "n/a" if pd.isna(v) else f"{v:.4f}", "best" if gap >= 0 else f"{gap:+.4f}", delta_color="normal" if gap >= 0 else "inverse")
        pred = Evaluator.predict(probs[self.model])
        left, right = st.columns(2)
        with left:
            st.markdown("**Confusion matrix**")
            cm = Evaluator.confusion(y, pred).rename_axis("Actual").reset_index().melt("Actual", var_name="Predicted", value_name="Count")
            base = alt.Chart(cm).encode(x=alt.X("Predicted:N", axis=alt.Axis(labelAngle=0)), y="Actual:N", tooltip=["Actual", "Predicted", "Count"])
            heat = base.mark_rect(cornerRadius=4).encode(color=alt.Color("Count:Q", scale=alt.Scale(range=["#231c45", "#a78bfa"]), legend=None))
            text = base.mark_text(fontSize=20, fontWeight="bold").encode(
                text="Count:Q", color=alt.condition(alt.datum.Count > cm.Count.max() / 2, alt.value("#0b1020"), alt.value("#e7eaf3")))
            st.altair_chart((heat + text).properties(height=280), width="stretch")
        with right:
            st.markdown("**Classification report**")
            st.dataframe(Evaluator.report(y, pred).style.format("{:.3f}").format("{:.0f}", subset=["support"]), width="stretch")

    def predictions(self, df: pd.DataFrame, probs: pd.DataFrame) -> None:
        out = pd.DataFrame({"Prediction": Evaluator.label(probs[self.model]), "P(malignant)": probs[self.model]}, index=df.index)
        if settings.target in df:
            out["Actual"] = df[settings.target].map(dict(enumerate(settings.labels)))
            out["Correct"] = out["Actual"] == out["Prediction"]
        show = st.segmented_control("Show", ["All", "Malignant", "Benign"], default="All", key="pred_filter") or "All"
        view = out if show == "All" else out[out["Prediction"] == show]
        st.dataframe(view, width="stretch", height=360, column_config={
            "P(malignant)": st.column_config.ProgressColumn(f"P(malignant) · {self.model}", min_value=0, max_value=1, format="percent"),
            "Correct": st.column_config.CheckboxColumn("Correct?"),
        })
        st.download_button("⬇️ Download predictions (all models)", df.join(probs.add_prefix("P_malignant_")).join(out[["Prediction"]]).to_csv(index=False), "predictions.csv", "text/csv")

    # ---------- single ----------
    def set_inputs(self, values: pd.Series) -> None:
        for f in self.service.features:
            st.session_state[f] = float(values[f])

    def single_tab(self) -> None:
        if self.service.features[0] not in st.session_state:
            self.set_inputs(self.medians)
        st.markdown("Enter the measurements for one patient, or start from an example:")
        a, b, c, _ = st.columns([1, 1, 1, 2])
        a.button("🟢 Benign example", on_click=self.set_inputs, args=(self.test_df[self.test_df[settings.target] == 0].iloc[0],), width="stretch")
        b.button("🔴 Malignant example", on_click=self.set_inputs, args=(self.test_df[self.test_df[settings.target] == 1].iloc[0],), width="stretch")
        c.button("↺ Reset to median", on_click=self.set_inputs, args=(self.medians,), width="stretch")

        bases = [f[5:] for f in self.service.features if f.startswith("mean ")]
        with st.form("patient"):
            for tab, pattern in zip(st.tabs(list(GROUPS)), GROUPS.values()):
                cols = tab.columns(5)
                for i, base in enumerate(bases):
                    cols[i % 5].number_input(base.capitalize(), min_value=0.0, format="%.4f", key=pattern.format(base))
            submitted = st.form_submit_button("🔍 Predict", type="primary", width="stretch")
        if submitted:
            st.session_state["record"] = {f: st.session_state[f] for f in self.service.features}
        if "record" in st.session_state:
            self.verdict(pd.DataFrame([st.session_state["record"]]))

    def verdict(self, record: pd.DataFrame) -> None:
        p = self.service.predict_proba(record).iloc[0]
        label = settings.labels[int(p[self.model] > settings.threshold)]
        votes = int(Evaluator.predict(p).sum())
        confidence = p[self.model] if label == "Malignant" else 1 - p[self.model]
        with st.container(border=True):
            left, right = st.columns([1, 2])
            left.markdown(
                f'<div class="verdict {label.lower()}"><div class="k">Prediction · {self.model}</div>'
                f'<div class="v">{"🔴" if label == "Malignant" else "🟢"} {label}</div><div class="row">'
                f'<div><span class="k">Confidence</span><b>{confidence:.1%}</b></div>'
                f'<div><span class="k">Models agreeing</span><b>{votes if label == "Malignant" else 5 - votes} / 5</b></div></div></div>',
                unsafe_allow_html=True,
            )
            with right:
                data = p.rename("p").rename_axis("Model").reset_index().assign(Verdict=lambda d: Evaluator.label(d.p))
                base = alt.Chart(data).encode(
                    x=alt.X("p:Q", title="P(malignant)", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                    y=alt.Y("Model:N", title=None, sort=list(self.service.models), axis=alt.Axis(labelLimit=200)),
                    tooltip=["Model", alt.Tooltip("p:Q", format=".1%", title="P(malignant)"), "Verdict"],
                )
                bars = base.mark_bar(cornerRadiusEnd=4).encode(
                    color=alt.Color("Verdict:N", scale=alt.Scale(domain=list(STATUS), range=list(STATUS.values())), legend=alt.Legend(orient="bottom", title=None)))
                text = base.mark_text(align="left", dx=4, color="#e7eaf3").encode(text=alt.Text("p:Q", format=".1%"))
                rule = alt.Chart(pd.DataFrame({"t": [settings.threshold]})).mark_rule(strokeDash=[4, 4], color=MUTED).encode(x="t:Q")
                st.markdown("**All models** · dashed line = 50% decision threshold")
                st.altair_chart((bars + text + rule).properties(height=240), width="stretch")
        st.caption("⚠️ Educational demo only. This is not a medical diagnosis.")


Dashboard().run()
