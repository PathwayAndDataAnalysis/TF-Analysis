import pandas as pd
from statsmodels.stats.multitest import multipletests


def main(p_value_file: str) -> pd.DataFrame:
    p_value = pd.read_csv(p_value_file, sep="\t")

    # Remove columns with all NaN values
    p_value.dropna(axis=1, how="all", inplace=True)

    # Make first column the index
    p_value.set_index("Unnamed: 0", inplace=True)

    # Create a dataframe of shape p_value with all NaN values
    df_reject = pd.DataFrame(index=p_value.index, columns=p_value.columns)

    # Benjamini-Hochberg
    for i in p_value.columns:
        tf_pval = p_value[i].dropna()

        reject, pvals_corrected, _, _ = multipletests(
            abs(tf_pval), alpha=0.05, method="fdr_bh"
        )

        tf_pval = pd.DataFrame(tf_pval)
        tf_pval["Reject"] = reject
        tf_pval["Corrected P-value"] = pvals_corrected

        df_reject[i] = tf_pval["Reject"]

    return df_reject


if __name__ == "__main__":
    p_file = "data/pvalue_3valid_target.tsv"
    bh_file = p_file.split(".")[0] + "_BH_REJECT.tsv"

    bh_output = main(p_file)
    bh_output.to_csv(bh_file, sep="\t")
