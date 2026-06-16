#Script use:

#python scripts/analyze_metrics.py \
#    --metrics outputs/out_test_10_water_5_19_26/metrics_pct_10_water_5_19_26.csv \
#    --outdir outputs/out_test_10_water_5_19_26/analysis

#OR

# python scripts/analyze_metrics.py \
#     --metrics \
#     outputs/out_test_10_water_5_19_26/metrics_pct_10_water_5_19_26.csv \
#     outputs/out_test_30_masako_effortful_5_18_26/metrics_pct_30_masako_effortful_5_18_26.csv \
#     outputs/out_test_25_water_5_25_26/metrics_pct_25_water_5_25_26.csv \
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

print("\nActivity counts:")
print(feature_df["activity"].value_counts())

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

coef_df = pd.DataFrame({
    "feature": features,
    "weight": lda.coef_.ravel()
})

print("\ncoef_df:")
print(coef_df.head())

coef_df.to_csv(
    outdir / "LDA_feature_importance.csv"
)


coef_df["abs_weight"] = np.abs(
    coef_df["weight"]
)

coef_df = coef_df.sort_values(
    "abs_weight",
    ascending=False
)

##AUC             ████████
##Amplitude       ██████
##FWHM            ████
##n_humps         ██

# This immediately tells me:

# Which physiological metrics are driving the separation.