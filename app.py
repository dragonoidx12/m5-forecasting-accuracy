
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).parent

st.set_page_config(
    page_title="M5 Sales Forecasting",
    page_icon="📈",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(
        ROOT / "m5_final_model.joblib"
    )

bundle = load_model()
features = bundle["feature_columns"]

st.title("📈 M5 Forecasting - Accuracy")

st.write(
    "Walmart ürün satışlarının 28 günlük "
    "tahmini için geliştirilmiş HGB modeli."
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Yerel WRMSSE",
    f"{bundle['local_wrmsse']:.6f}"
)

c2.metric("Tahmin ufku", "28 gün")
c3.metric("Model özellikleri", len(features))

st.info(
    "WRMSSE değeri yerel tarihsel doğrulamadan "
    "gelmektedir; Kaggle leaderboard skoru değildir."
)

st.warning(
    "Eğitim amaçlı uygulamadır. Örnek simülasyon "
    "gerçek mağaza veya ürün verisi değildir. "
    "Ham M5 satış dosyaları doğrudan kabul edilmez."
)

mode = st.selectbox(
    "Tahmin dönemi",
    [
        "Validation (d_1914 - d_1941)",
        "Evaluation (d_1942 - d_1969)"
    ]
)

if mode.startswith("Validation"):
    model = bundle["validation_model"]
    demo_file = "demo_validation.csv"
else:
    model = bundle["evaluation_model"]
    demo_file = "demo_evaluation.csv"

tab1, tab2 = st.tabs([
    "28 Günlük Simülasyon",
    "CSV ile Tahmin"
])

with tab1:
    st.subheader("Yapay satış senaryosu")

    st.caption(
        "Başlangıç senaryosu: 168 gün boyunca "
        "günde 1 adet satış. Diğer anonim ürün "
        "ve mağaza kodları temsili değerlerdir."
    )

    intensity = st.slider(
        "Geçmiş satış yoğunluğu çarpanı",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.25
    )

    X = pd.read_csv(ROOT / demo_file)

    volume_features = [
        "lag_1", "lag_7", "lag_14", "lag_28",
        "mean_7", "mean_28",
        "mean_56", "mean_168",
        "std_28", "max_28",
        "sum_28", "trend_7_28",
        "same_weekday_1", "same_weekday_2",
        "same_weekday_3", "same_weekday_4",
        "same_weekday_mean"
    ]

    for col in volume_features:
        X[col] *= intensity

    if st.button(
        "28 Günlük Satışları Tahmin Et",
        type="primary"
    ):
        prediction = np.maximum(
            model.predict(X[features]),
            0
        )

        chart = pd.DataFrame({
            "Gün": np.arange(1, 29),
            "Tahmini Satış": prediction
        }).set_index("Gün")

        st.line_chart(chart)

        st.metric(
            "28 günlük tahmini toplam satış",
            f"{prediction.sum():.2f} adet"
        )

        st.dataframe(chart)

        st.download_button(
            "Tahminleri CSV olarak indir",
            data=chart.to_csv(),
            file_name="m5_demo_forecast.csv",
            mime="text/csv"
        )

with tab2:
    st.subheader("Mühendisliği yapılmış özellik CSV'si")

    st.write(
        f"Her satır için {len(features)} özellik "
        "içeren bir CSV yükleyebilirsin."
    )

    with st.expander("Gerekli özellikleri göster"):
        st.code(", ".join(features))

    uploaded = st.file_uploader(
        "Özellik CSV dosyası",
        type=["csv"]
    )

    if uploaded is not None:
        df = pd.read_csv(uploaded, nrows=10001)

        missing = [
            col for col in features
            if col not in df.columns
        ]

        if missing:
            st.error(
                "Eksik sütunlar: " + ", ".join(missing)
            )

        elif not 1 <= len(df) <= 10000:
            st.error(
                "En fazla 10.000 satır yükleyebilirsin."
            )

        else:
            numeric = df[features].apply(
                pd.to_numeric, errors="coerce"
            )

            if not np.isfinite(
                numeric.to_numpy(dtype=float)
            ).all():
                st.error(
                    "Eksik veya sayısal olmayan özellik var."
                )

            elif st.button(
                "CSV Tahminlerini Hesapla",
                type="primary"
            ):
                preds = np.maximum(
                    model.predict(numeric),
                    0
                )

                result = pd.DataFrame({
                    "row_number": np.arange(
                        1, len(preds) + 1
                    ),
                    "forecast_sales": preds
                })

                st.dataframe(result.head(50))

                st.download_button(
                    "CSV sonuçlarını indir",
                    data=result.to_csv(index=False),
                    file_name="m5_batch_predictions.csv",
                    mime="text/csv"
                )

st.caption(
    "Kaggle M5 Forecasting - Accuracy | "
    "Educational machine learning project"
)
