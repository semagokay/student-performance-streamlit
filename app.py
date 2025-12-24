import io
import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# -----------------------
# Page setup
# -----------------------
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.title("Higher Education Students Performance — Streamlit Dashboard")

st.caption(
    "Bu dashboard, hocanızın Streamlit exercise’ındaki EDA + PCA + ML akışını "
    "Higher Education Students Performance veri setine uyarlamak için hazırlanmıştır."
)

# -----------------------
# Load data
# -----------------------
st.sidebar.header("1) Veri Yükleme")
uploaded = st.sidebar.file_uploader("CSV yükle (opsiyonel)", type=["csv"])

@st.cache_data(show_spinner=False)
def load_df_from_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b))

@st.cache_data(show_spinner=False)
def load_df_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# Default: local data.csv (repo içine koyulmalı)
DEFAULT_PATH = "data.csv"

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.sidebar.success("CSV yüklendi.")
else:
    try:
        df = load_df_from_path(DEFAULT_PATH)
        st.sidebar.info("Repo içindeki data.csv kullanılıyor.")
    except Exception as e:
        st.error(
            "data.csv bulunamadı. Sol panelden CSV yükleyin veya repo içine data.csv ekleyin.\n\n"
            f"Hata: {e}"
        )
        st.stop()

# -----------------------
# Basic cleaning / types
# -----------------------
# UCI dataset often has numbered columns "1".."30". Keep them as strings.
df.columns = [str(c).strip() for c in df.columns]

# Optional filters
st.sidebar.header("2) Filtreler")
course_col = "COURSE ID" if "COURSE ID" in df.columns else None
grade_col = "GRADE" if "GRADE" in df.columns else None

if course_col:
    course_vals = sorted(df[course_col].dropna().unique().tolist())
    selected_courses = st.sidebar.multiselect("COURSE ID filtrele", course_vals, default=course_vals)
    df_view = df[df[course_col].isin(selected_courses)].copy()
else:
    df_view = df.copy()

st.sidebar.divider()

# -----------------------
# 1) Head / overview
# -----------------------
st.subheader("1) İlk 10 satır")
st.dataframe(df_view.head(10), use_container_width=True)

st.subheader("2) Dataset özeti")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Satır", df_view.shape[0])
c2.metric("Sütun", df_view.shape[1])
c3.metric("Eksik değer", int(df_view.isna().sum().sum()))
c4.metric("COURSE sayısı", int(df_view[course_col].nunique()) if course_col else 0)

with st.expander("Sütun tipleri (dtype)"):
    st.dataframe(
        df_view.dtypes.astype(str).reset_index().rename(columns={"index": "column", 0: "dtype"}),
        use_container_width=True
    )

st.divider()

# -----------------------
# Auto-detect categorical vs numeric (based on unique count)
# -----------------------
def is_categorical(series: pd.Series) -> bool:
    # If object -> categorical; if numeric but small unique -> treat as categorical (encoded categories)
    if series.dtype == "object":
        return True
    nunique = series.nunique(dropna=True)
    return nunique <= 10

cat_cols = [c for c in df_view.columns if is_categorical(df_view[c]) and c != "STUDENT ID"]
num_cols = [c for c in df_view.columns if c not in cat_cols and c != "STUDENT ID"]

# Many columns are encoded categories; keep the target in its own group
if grade_col and grade_col in cat_cols:
    cat_cols = [c for c in cat_cols if c != grade_col]
elif grade_col and grade_col in num_cols:
    num_cols = [c for c in num_cols if c != grade_col]

# -----------------------
# 3) Categorical distribution (Pie / Bar)
# -----------------------
st.subheader("3) Kategorik değişken analizi (Pie / Bar)")
if len(cat_cols) == 0:
    st.info("Kategorik sütun bulunamadı (veya hepsi sayısal görünüyor).")
else:
    left, right = st.columns([1, 2])
    with left:
        cat = st.selectbox("Kategorik sütun seç", cat_cols)
        topn = st.slider("En fazla kaç kategori gösterilsin?", 5, 30, 10)
        chart_type = st.radio("Grafik tipi", ["Pie", "Bar"], horizontal=True)
    vc = df_view[cat].fillna("NA").value_counts().head(topn)

    fig, ax = plt.subplots()
    if chart_type == "Pie":
        ax.pie(vc.values, labels=[str(x) for x in vc.index], autopct="%1.1f%%")
        ax.set_title(f"{cat} dağılımı (Top {topn})")
    else:
        ax.bar([str(x) for x in vc.index], vc.values)
        ax.set_title(f"{cat} dağılımı (Top {topn})")
        ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig, use_container_width=True)

st.divider()

# -----------------------
# 4) Top 10 group by chosen category (count)
# -----------------------
st.subheader("4) Top 10 kategori (işlem/satır sayısı)")
if len(cat_cols) == 0:
    st.info("Bu bölüm için kategorik sütun gerekli.")
else:
    group_col = st.selectbox("Gruplama sütunu", cat_cols, index=0)
    top = df_view[group_col].fillna("NA").value_counts().head(10)

    fig, ax = plt.subplots()
    ax.bar([str(x) for x in top.index], top.values)
    ax.set_title(f"Top 10 {group_col} (count)")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig, use_container_width=True)

st.divider()

# -----------------------
# 5) Scatter plot (any two numeric-like columns)
# -----------------------
st.subheader("5) Scatter Plot (İki değişken ilişkisi)")
# Treat all non-object columns as numeric candidates (even if encoded)
numeric_candidates = [c for c in df_view.columns if c != "STUDENT ID"]
if grade_col and grade_col in numeric_candidates:
    # keep grade available for y-axis too; don't remove
    pass

if len(numeric_candidates) < 2:
    st.info("Scatter için en az 2 sayısal/encode sütun gerekli.")
else:
    left, right = st.columns(2)
    with left:
        xcol = st.selectbox("X sütunu", numeric_candidates, index=0)
    with right:
        ycol = st.selectbox("Y sütunu", numeric_candidates, index=min(1, len(numeric_candidates)-1))

    x = pd.to_numeric(df_view[xcol], errors="coerce")
    y = pd.to_numeric(df_view[ycol], errors="coerce")
    mask = x.notna() & y.notna()

    fig, ax = plt.subplots()
    ax.scatter(x[mask], y[mask], s=18)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(f"{xcol} vs {ycol}")
    st.pyplot(fig, use_container_width=True)

st.divider()

# -----------------------
# 6) Derived score + histogram
# -----------------------
st.subheader("6) Türetilmiş skor: TotalScore (1..30 sütunlarının toplamı)")
feature_num_cols = [c for c in df_view.columns if re.fullmatch(r"\d+", c)]
if len(feature_num_cols) > 0:
    tmp = df_view.copy()
    for c in feature_num_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
    tmp["TotalScore"] = tmp[feature_num_cols].sum(axis=1, skipna=True)

    fig, ax = plt.subplots()
    ax.hist(tmp["TotalScore"].dropna(), bins=25)
    ax.set_title("TotalScore dağılımı")
    ax.set_xlabel("TotalScore")
    st.pyplot(fig, use_container_width=True)

    if grade_col and grade_col in df_view.columns:
        # Boxplot: TotalScore by Grade
        st.caption("TotalScore ile GRADE ilişkisi (boxplot)")
        grades = sorted(df_view[grade_col].dropna().unique().tolist())
        data = [tmp.loc[tmp[grade_col] == g, "TotalScore"].dropna().values for g in grades]
        fig2, ax2 = plt.subplots()
        ax2.boxplot(data, labels=[str(g) for g in grades])
        ax2.set_title("TotalScore by GRADE")
        ax2.set_xlabel("GRADE")
        ax2.set_ylabel("TotalScore")
        st.pyplot(fig2, use_container_width=True)
else:
    st.info("Bu veri setinde '1..30' formatında sütunlar bulunamadı, TotalScore üretilemedi.")

st.divider()

# -----------------------
# 7) "Time series" alternative (since no date)
# Instead: distribution by Course ID (stacked)
# -----------------------
st.subheader("7) Zaman serisi yoksa alternatif: COURSE ID bazında GRADE dağılımı")
if course_col and grade_col and course_col in df_view.columns and grade_col in df_view.columns:
    pivot = pd.crosstab(df_view[course_col], df_view[grade_col])
    st.dataframe(pivot, use_container_width=True)
else:
    st.info("COURSE ID ve GRADE sütunları yoksa bu bölüm atlanır.")

st.divider()

# -----------------------
# 8) Correlation heatmap (numeric)
# -----------------------
st.subheader("8) Korelasyon Heatmap (sayısal sütunlar)")
# Use numeric conversion for all columns except STUDENT ID
num_df = df_view.drop(columns=["STUDENT ID"], errors="ignore").copy()
for c in num_df.columns:
    num_df[c] = pd.to_numeric(num_df[c], errors="coerce")

num_df = num_df.dropna(axis=0, how="any")
if num_df.shape[0] < 10:
    st.info("Korelasyon için yeterli temiz satır yok.")
else:
    corr = num_df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(corr.values)
    ax.set_title("Correlation Matrix")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig, use_container_width=True)

st.divider()

# -----------------------
# 9) PCA (2D) + coloring by Grade
# -----------------------
st.subheader("9) PCA (2 bileşen) — 2D görselleştirme")
pca_cols_default = feature_num_cols[:10] if len(feature_num_cols) >= 10 else feature_num_cols
if len(pca_cols_default) < 2:
    st.info("PCA için en az 2 feature gerekli.")
else:
    pca_cols = st.multiselect("PCA için kullanılacak sütunlar", feature_num_cols, default=pca_cols_default)
    if len(pca_cols) < 2:
        st.warning("En az 2 sütun seçmelisin.")
    else:
        X = df_view[pca_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if X.shape[0] < 20:
    st.warning("PCA için yeterli temiz satır yok. Farklı sütun seç veya filtreleri azalt.")
    st.stop()
        else:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X.values)

            pca = PCA(n_components=2, random_state=42)
            comps = pca.fit_transform(Xs)

            fig, ax = plt.subplots()
            if grade_col and grade_col in df_view.columns:
                # Align grade with X index
                g = df_view.loc[X.index, grade_col].astype(str).values
                # Simple coloring by grade using multiple scatters (no explicit colors set)
                for gg in np.unique(g):
                    m = (g == gg)
                    ax.scatter(comps[m, 0], comps[m, 1], s=18, label=f"GRADE {gg}")
                ax.legend(fontsize=8, ncols=2)
            else:
                ax.scatter(comps[:, 0], comps[:, 1], s=18)

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"PCA 2D (Explained var: {pca.explained_variance_ratio_.sum():.2f})")
            st.pyplot(fig, use_container_width=True)

st.divider()

# -----------------------
# 10) KMeans clustering on PCA space (segmentasyon)
# -----------------------
st.subheader("10) Segmentasyon: KMeans (PCA uzayında)")
if len(pca_cols_default) >= 2:
    # reuse pca_cols from above if exists
    if "pca_cols" in locals() and len(pca_cols) >= 2:
        X = df_view[pca_cols].apply(pd.to_numeric, errors="coerce").dropna()
        if X.shape[0] >= 20:
            Xs = StandardScaler().fit_transform(X.values)
            comps = PCA(n_components=2, random_state=42).fit_transform(Xs)
            
            if comps.shape[0] < 20:
            st.warning("KMeans için yeterli veri yok.")
            st.stop()

            k = st.slider("Cluster sayısı (k)", 2, 6, 3)
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(comps)

            fig, ax = plt.subplots()
            for lab in np.unique(labels):
                m = (labels == lab)
                ax.scatter(comps[m, 0], comps[m, 1], s=18, label=f"Cluster {lab}")
            ax.legend(fontsize=8)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("KMeans clusters (PCA 2D)")
            st.pyplot(fig, use_container_width=True)

            st.caption("Cluster özetleri (feature ortalamaları):")
            tmp = df_view.loc[X.index, pca_cols].copy()
            tmp["cluster"] = labels
            st.dataframe(tmp.groupby("cluster").mean(numeric_only=True), use_container_width=True)
        else:
            st.info("KMeans için yeterli veri yok.")
    else:
        st.info("Önce PCA sütunlarını seç.")
else:
    st.info("KMeans için PCA yapılabilecek feature yok.")

st.divider()

# -----------------------
# 11) ML: Model comparison (LogReg vs RandomForest) predicting GRADE
# -----------------------
st.subheader("11) Makine Öğrenmesi: GRADE tahmini (Model karşılaştırma)")
if not grade_col or grade_col not in df_view.columns:
    st.info("Bu bölüm için GRADE sütunu gerekli.")
else:
    # Features: use the numbered 1..30 columns if available, else all numeric columns except target
    if len(feature_num_cols) >= 2:
        X_all = df_view[feature_num_cols].apply(pd.to_numeric, errors="coerce")
    else:
        X_all = df_view.drop(columns=["STUDENT ID", grade_col], errors="ignore").apply(pd.to_numeric, errors="coerce")

    y_all = pd.to_numeric(df_view[grade_col], errors="coerce")

    mask = X_all.notna().all(axis=1) & y_all.notna()
    X = X_all[mask]
    y = y_all[mask].astype(int)

    if len(X) < 50:
        st.warning("Model eğitimi için temiz veri az olabilir (>=50 önerilir).")

    test_size = st.slider("Test oranı", 0.1, 0.4, 0.2, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=test_size, random_state=42, stratify=y.values if len(np.unique(y.values)) > 1 else None
    )

    # Logistic Regression (multinomial)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=2000, multi_class="auto")
    lr.fit(X_train_s, y_train)
    pred_lr = lr.predict(X_test_s)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=400, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    # Metrics
    acc_lr = accuracy_score(y_test, pred_lr)
    acc_rf = accuracy_score(y_test, pred_rf)
    f1_lr = f1_score(y_test, pred_lr, average="macro")
    f1_rf = f1_score(y_test, pred_rf, average="macro")

    m1, m2 = st.columns(2)
    with m1:
        st.write("**Logistic Regression**")
        st.write(f"Accuracy: {acc_lr:.3f}")
        st.write(f"Macro F1: {f1_lr:.3f}")
    with m2:
        st.write("**Random Forest**")
        st.write(f"Accuracy: {acc_rf:.3f}")
        st.write(f"Macro F1: {f1_rf:.3f}")

    # Confusion Matrix for best model
    best_name, best_pred = ("Random Forest", pred_rf) if f1_rf >= f1_lr else ("Logistic Regression", pred_lr)
    cm = confusion_matrix(y_test, best_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(cm)
    ax.set_title(f"Confusion Matrix — {best_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))
    st.pyplot(fig, use_container_width=True)

    # Feature importance (RF)
    if hasattr(rf, "feature_importances_"):
        st.subheader("12) Feature Importance (Random Forest)")
        importances = pd.Series(rf.feature_importances_, index=X.columns.astype(str)).sort_values(ascending=False).head(15)
        fig, ax = plt.subplots()
        ax.bar(importances.index.astype(str), importances.values)
        ax.set_title("Top 15 Feature Importances")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig, use_container_width=True)

st.divider()

# -----------------------
# 13) Short interpretation block
# -----------------------
st.subheader("13) Kısa yorum (sunum için)")
st.write(
    """
- **EDA** bölümünde dağılımlar ve ilişkiler görüldü (kategorik pie/bar + scatter + histogram).
- **TotalScore**, öğrenci davranış/tercih değişkenlerinin genel bir özetini verir; GRADE ile ilişkisi boxplot ile gözlenebilir.
- **PCA**, çok boyutlu veriyi 2 boyuta indirip örüntüleri (benzer öğrenci profilleri) görselleştirir.
- **KMeans**, benzer öğrenci profillerini kümelere ayırarak segmentasyon sağlar.
- **ML** bölümünde GRADE tahmini yapılıp iki model (LogReg vs RandomForest) karşılaştırılır.
"""
)
