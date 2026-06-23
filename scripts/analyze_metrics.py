#Script use:

#python scripts/analyze_metrics.py \
#    --metrics outputs/out_test_10_water_5_19_26/metrics_pct_10_water_5_19_26.csv \
#    --outdir outputs/out_test_10_water_5_19_26/analysis

#OR

# python scripts/analyze_metrics.py \
#     --metrics \
#     outputs/out_test_10_water_5_19_26/metrics_pct_10_water_5_19_26.csv \
#     outputs/out_test_30_masako_effortful_5_18_26/metrics_pct_30_masako_effortful_5_18_26.csv \
#     outputs/out_test_10_water_5_28_26/metrics_pct_10_water_5_28_26.csv \
#     --outdir outputs/combined_analysis

#python scripts/analyze_metrics.py \
#    --metrics outputs/out_test_30_masako_effortful_5_18_26/metrics_pct_30_masako_effortful_5_18_26.csv \
#    --outdir outputs/out_test_30_masako_effortful_5_18_26/analysis

#ALL masako and effortful trials can be run with:

# python scripts/analyze_metrics.py \
#     --metrics \
#     outputs/out_test_30_masako_effortful_5_18_26/metrics_pct_30_masako_effortful_5_18_26.csv \
#     outputs/out_test_30_masako_effortful_5_27_26/metrics_pct_30_masako_effortful_5_27_26.csv \
#     outputs/out_test_40_masako_effortful_6_15_26/metrics_pct_40_masako_effortful_6_15_26.csv \
#     --outdir outputs/combined_analysis_masako_and_effortful_100



    #########################

# python scripts/analyze_metrics.py \
#     --metrics outputs/out_test_30_masako_effortful_5_27_26/metrics_pct_30_masako_effortful_5_27_26.csv \
#     --outdir outputs/out_test_30_masako_effortful_5_27_26/analysis

# python scripts/analyze_metrics.py \
#     --metrics outputs/out_test_40_masako_effortful_6_15_26/metrics_pct_40_masako_effortful_6_15_26.csv \
#     --outdir outputs/out_test_40_masako_effortful_6_15_26/analysis


#to run all water, masako, and effortful trial metrics data together:
# python scripts/analyze_metrics.py \
#     --metrics \
#     outputs/out_test_30_masako_effortful_5_18_26/metrics_pct_30_masako_effortful_5_18_26.csv \
#     outputs/out_test_30_masako_effortful_5_27_26/metrics_pct_30_masako_effortful_5_27_26.csv \
#     outputs/out_test_40_masako_effortful_6_15_26/metrics_pct_40_masako_effortful_6_15_26.csv \
#     outputs/out_test_10_water_5_19_26/metrics_pct_10_water_5_19_26.csv \
#     outputs/out_test_10_water_5_20_26/metrics_pct_10_water_5_20_26.csv \
#     outputs/out_test_10_water_5_28_26/metrics_pct_10_water_5_28_26.csv \
#     outputs/out_test_10_water_6_5_26/metrics_pct_10_water_6_5_26.csv \
#     --outdir outputs/combined_analysis_water_masako_effortful_100

# python3 scripts/analyze_metrics.py \
#     --metrics \
#     outputs/out_test_effortful_swallow_fail_6_23_26/metrics_pct_effortful_swallow_fail_6_23_26.csv \
#     outputs/out_test_30_masako_effortful_5_18_26/metrics_pct_30_masako_effortful_5_18_26.csv \
#     --outdir outputs/out_test_effortful_swallow_fail_6_23_26/analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument(
    "--metrics",
    nargs="+",
    required=True,
    help="One or more metrics_pct CSV files from process_effortful pipeline"
)

parser.add_argument(
    "--outdir",
    required=True,
    help="output folder"
)

args = parser.parse_args()

metrics_files = args.metrics
outdir = Path(args.outdir)

outdir.mkdir(parents=True, exist_ok=True)


####################


#mm = pd.read_csv(metrics_csv)

all_metrics = []

for f in metrics_files:

    print(f"Loading: {f}")

    tmp = pd.read_csv(f)

    tmp["source_file"] = Path(f).stem

    all_metrics.append(tmp)

mm = pd.concat(
    all_metrics,
    ignore_index=True
)

mean_metrics = mm[mm["channel"] == "mean"]

print("\nColumns available:")
print(mean_metrics.columns.tolist())

features = [
    "amplitude_pct",
    "auc_pct",
    "fwhm_seconds",
    "n_humps",
    "mean_inter_gulp_s",
    "mean_time_to_peak_s",
    "mean_time_to_dip_s",
]

X = mean_metrics[features]
y = mean_metrics["gulp"]


feature_df = mean_metrics.copy()

def activity_type(g):
    g = str(g).lower()

    if "effortful swallow" in g:
        return "effortful"

    if "masako maneuver" in g:
        return "masako"

    if "3oz water" in g:
        return "water"

    return "unknown"

feature_df["activity"] = feature_df["gulp"].apply(activity_type)

if "pass_fail" in feature_df.columns:
    feature_df["pass_fail"] = feature_df["pass_fail"].astype(str).str.lower().replace({"nan": "pass"})
else:
    def pass_fail_label(gulp):
        g = str(gulp).lower()
        if "fail" in g:
            return "fail"
        return "pass"

    feature_df["pass_fail"] = feature_df["gulp"].apply(pass_fail_label)

print("\nActivity counts:")
print(feature_df["activity"].value_counts())

print("\nPass/Fail counts:")
print(feature_df["pass_fail"].value_counts())

effortful_df = feature_df[
    feature_df["activity"] == "effortful"
].copy()

X_eff = effortful_df[features].copy()
X_eff = X_eff.replace([np.inf, -np.inf], np.nan).dropna()
y_eff = effortful_df.loc[X_eff.index, "pass_fail"]

if y_eff.nunique() > 1:
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    scaler_eff = StandardScaler()
    X_eff_scaled = scaler_eff.fit_transform(X_eff)

    pca2 = PCA(n_components=2)
    X_eff_pca = pca2.fit_transform(X_eff_scaled)

    pca_eff_df = pd.DataFrame(
        X_eff_pca,
        columns=["PC1", "PC2"],
        index=X_eff.index,
    )
    pca_eff_df["pass_fail"] = y_eff.values
    pca_eff_df.to_csv(
        outdir / "PCA_effortful_pass_fail_coordinates.csv",
        index=False,
    )

    plt.figure(figsize=(8,6))
    for label in sorted(pca_eff_df["pass_fail"].unique()):
        mask = pca_eff_df["pass_fail"] == label
        plt.scatter(
            pca_eff_df.loc[mask, "PC1"],
            pca_eff_df.loc[mask, "PC2"],
            label=label,
            alpha=0.8,
        )
    plt.xlabel(f"PC1 ({100*pca2.explained_variance_ratio_[0]:.1f}%)")
    plt.ylabel(f"PC2 ({100*pca2.explained_variance_ratio_[1]:.1f}%)")
    plt.title("PCA: Effortful Swallow Pass vs Fail")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "PCA_effortful_pass_fail.png", dpi=150)
    plt.close()

    lda2 = LinearDiscriminantAnalysis(n_components=1)
    X_eff_lda = lda2.fit_transform(X_eff_scaled, y_eff)

    lda_eff_df = pd.DataFrame({
        "LD1": X_eff_lda[:, 0],
        "pass_fail": y_eff.values,
    })
    lda_eff_df.to_csv(
        outdir / "LDA_effortful_pass_fail_coordinates.csv",
        index=False,
    )

    plt.figure(figsize=(8,4))
    for label in sorted(lda_eff_df["pass_fail"].unique()):
        mask = lda_eff_df["pass_fail"] == label
        plt.scatter(
            lda_eff_df.loc[mask, "LD1"],
            np.zeros(mask.sum()),
            label=label,
            alpha=0.8,
        )
    plt.xlabel("LD1")
    plt.yticks([])
    plt.title("LDA: Effortful Swallow Pass vs Fail")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "LDA_effortful_pass_fail.png", dpi=150)
    plt.close()
else:
    print("\nSkipping effortful pass/fail PCA/LDA: need at least two pass_fail classes.")


plt.figure(figsize=(8,6))

feature_df.boxplot(
    column="amplitude_pct",
    by="activity"
)

plt.ylabel("Amplitude (%)")
plt.title("Amplitude by Activity")
plt.suptitle("")
plt.tight_layout()

plt.savefig(outdir / "boxplot_amplitude_by_activity.png", dpi=150)
plt.close()

plt.figure(figsize=(8,6))

feature_df.boxplot(
    column="auc_pct",
    by="activity"
)

plt.ylabel("AUC (%·s)")
plt.title("AUC by Activity")
plt.suptitle("")
plt.tight_layout()

plt.savefig(outdir / "boxplot_auc_by_activity.png", dpi=150)
plt.close()


plt.figure(figsize=(8,6))

feature_df.boxplot(
    column="fwhm_seconds",
    by="activity"
)

plt.ylabel("FWHM (s)")
plt.title("FWHM by Activity")
plt.suptitle("")
plt.tight_layout()

plt.savefig(outdir / "boxplot_fwhm_by_activity.png", dpi=150)
plt.close()


features = [
    "amplitude_pct",
    "auc_pct",
    "fwhm_seconds",
    "n_humps",
    "mean_time_to_peak_s",
    "mean_time_to_dip_s"
]

from sklearn.preprocessing import StandardScaler

missing = [c for c in features if c not in feature_df.columns]

if missing:
    raise ValueError(
        f"Missing columns in metrics CSV: {missing}"
    )

X = feature_df[features].copy()

X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna()
valid_idx = X.index


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nPCA explained variance:")
print(pca.explained_variance_ratio_)
print("Total:",
      pca.explained_variance_ratio_.sum())


pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

pca_df.to_csv(
    outdir / "PCA_coordinates.csv",
    index=False
)



pca_df["activity"] = feature_df.loc[X.index, "activity"].values

plt.figure(figsize=(8,6))

for activity in pca_df["activity"].unique():

    mask = pca_df["activity"] == activity

    plt.scatter(
        pca_df.loc[mask, "PC1"],
        pca_df.loc[mask, "PC2"],
        label=activity
    )

plt.xlabel(
    f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)"
)

plt.ylabel(
    f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)"
)

plt.legend()
plt.title("PCA of Swallow Features")

plt.tight_layout()

plt.savefig(outdir / "PCA_activity_types.png", dpi=150)
plt.close()

loading_df = pd.DataFrame(
    pca.components_.T,
    index=features,
    columns=["PC1","PC2"]
)

loading_df.to_csv(
    outdir / "PCA_feature_loadings.csv"
)

import seaborn as sns

plt.figure(figsize=(6,4))

sns.heatmap(
    loading_df,
    annot=True,
    cmap="coolwarm",
    center=0
)

plt.title("PCA Feature Loadings")

plt.tight_layout()

plt.savefig(
    outdir / "PCA_feature_loadings_heatmap.png",
    dpi=150
)

plt.close()

y = feature_df.loc[valid_idx, "activity"]


corr = feature_df[features].corr()
plt.figure(figsize=(7,6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    center=0,
    square=True
)

plt.title("Feature Correlation Matrix")

plt.tight_layout()

plt.savefig(
    outdir / "feature_correlation_matrix.png",
    dpi=150
)

plt.close()



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

n_classes = len(np.unique(y))

lda = LinearDiscriminantAnalysis(
    n_components=min(2, n_classes - 1)
)

X_lda = lda.fit_transform(
    X_scaled,
    y
)

if X_lda.shape[1] == 1:

    lda_df = pd.DataFrame({
        "LD1": X_lda[:,0],
        "activity": y.values
    })

else:

    lda_df = pd.DataFrame(
        X_lda,
        columns=["LD1","LD2"]
    )

    lda_df["activity"] = y.values

lda_df.to_csv(
    outdir / "LDA_coordinates.csv",
    index=False
)

plt.figure(figsize=(8,6))

if "LD2" in lda_df.columns:

    for activity in lda_df["activity"].unique():

        mask = lda_df["activity"] == activity

        plt.scatter(
            lda_df.loc[mask, "LD1"],
            lda_df.loc[mask, "LD2"],
            label=activity
        )

    plt.xlabel("LD1")
    plt.ylabel("LD2")

else:

    for activity in lda_df["activity"].unique():

        mask = lda_df["activity"] == activity

        plt.scatter(
            lda_df.loc[mask, "LD1"],
            np.zeros(mask.sum()),
            label=activity
        )

    plt.xlabel("LD1")
    plt.yticks([])

plt.legend()

plt.title("LDA Activity Separation")

plt.tight_layout()

plt.savefig(
    outdir / "LDA_activity_types.png",
    dpi=150
)

plt.close()


# when i get pass/fail data, change:
# y = feature_df["activity"] to
# y = feature_df["pass_fail"] where pass and fail are my labels
# Then LDA becomes:

# Can these optical features distinguish healthy vs abnormal swallowing?

# which is my clinically important question.


#update:
#Once you collect failed trials, don't replace activity with pass/fail.

#Keep BOTH.

#Create:

#activity
#pass_fail

# columns.

# Then run:

# PCA #1
# color = activity

# Question:

# Can effortful, masako, and water be separated?

# PCA #2
# color = pass_fail

# Question:

# Can healthy and abnormal trials be separated?

# LDA #1
# y = activity

# Question:

# Which features distinguish swallowing tasks?

# LDA #2
# y = pass_fail

# Question:

# Which features distinguish healthy vs abnormal execution?

# The second one is likely much more clinically valuable for a publication.
#
#
#

#feature importance:

# coef_df = pd.DataFrame({
#     "feature": features,
#     "weight": lda.coef_[0]
# })

# coef_df.to_csv(
#     outdir / "LDA_feature_importance.csv",
#     index=False
# )

# coef_df = pd.DataFrame(
#     lda.scalings_,
#     index=features
# )

# coef_df.to_csv(
#     outdir / "LDA_feature_importance.csv"
#)

print("\nLDA coefficient shape:")
print(lda.coef_.shape)

# coef_df = pd.DataFrame({
#     "feature": features,
#     "weight": lda.coef_.ravel()
# })

# class-wise coefifcients:
coef_df = pd.DataFrame(
    lda.coef_,
    columns=features
)

coef_df["class"] = lda.classes_
coef_df.to_csv(outdir / "LDA_feature_importance_by_class.csv", index=False)

print("\ncoef_df:")
print(coef_df.head())

# --- global importance (FIXED VERSION) ---
global_importance = pd.DataFrame({
    "feature": features,
    "mean_abs_weight": np.mean(np.abs(lda.coef_), axis=0)
}).sort_values("mean_abs_weight", ascending=False)

global_importance.to_csv(
    outdir / "LDA_feature_importance_global.csv",
    index=False
)

print("\nGlobal feature importance:")
print(global_importance)


# coef_df.to_csv(
#     outdir / "LDA_feature_importance.csv"
# )


# coef_df["abs_weight"] = np.abs(
#     coef_df["weight"]
# )

# coef_df = coef_df.sort_values(
#     "abs_weight",
#     ascending=False
# )

print("\nLDA feature ranking:")
print(coef_df)

##AUC             ████████
##Amplitude       ██████
##FWHM            ████
##n_humps         ██

# This immediately tells me:

# Which physiological metrics are driving the separation.


from scipy.stats import mannwhitneyu

eff = feature_df[
    feature_df["activity"]=="effortful"
]

mas = feature_df[
    feature_df["activity"]=="masako"
]

for f in features:

    stat,p = mannwhitneyu(
        eff[f],
        mas[f]
    )

    print(f"{f}: p={p:.4e}")

#########

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ----------------------------
# Random Forest model
# ----------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

# ----------------------------
# 1. Cross-validated accuracy (most important baseline)
# ----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    rf,
    X_scaled,
    y,
    cv=cv,
    scoring="accuracy"
)

print("\nRandom Forest CV Accuracy:")
print("Mean:", cv_scores.mean())
print("Std:", cv_scores.std())

# ----------------------------
# 2. Train/test split evaluation (for ROC + confusion matrix later)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------
# 3. Confusion matrix plot
# ----------------------------
cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=np.unique(y),
    yticklabels=np.unique(y)
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Random Forest Confusion Matrix")

plt.tight_layout()
plt.savefig(outdir / "RF_confusion_matrix.png", dpi=150)
plt.close()

# ----------------------------
# 4. Feature importance
# ----------------------------
rf_importance = pd.DataFrame({
    "feature": features,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

rf_importance.to_csv(
    outdir / "RF_feature_importance.csv",
    index=False
)

print("\nRandom Forest Feature Importance:")
print(rf_importance)

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# ----------------------------
# Ensure binary classification for ROC
# ----------------------------
unique_classes = np.unique(y)

if len(unique_classes) != 2:
    print("\nROC skipped: not a binary classification problem.")
else:

    # Fit model (if not already fit above)
    rf.fit(X_train, y_train)

    # Predict probabilities for positive class
    y_proba = rf.predict_proba(X_test)[:, 1]

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba, pos_label=unique_classes[1])
    roc_auc = auc(fpr, tpr)

    print("\nROC AUC:", roc_auc)

    # ----------------------------
    # Plot ROC curve
    # ----------------------------
    plt.figure(figsize=(6,6))

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Random Forest ROC Curve")

    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(outdir / "RF_ROC_curve.png", dpi=150)
    plt.close()