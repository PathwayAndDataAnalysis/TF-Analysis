import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
import requests
import sys
from joblib import Parallel, delayed


def distribution_worker(max_target: int, ranks: np.array):
    arr = np.zeros(max_target)
    cs = np.random.choice(ranks, max_target, replace=False).cumsum()
    for i in range(0, max_target):
        amr = cs[i] / (i + 1)
        arr[i] = np.min([amr, 1 - amr])
    return arr


def get_distribution(max_target: int, total_genes: int, iters: int):
    dist_file = f"data/dist_{max_target}_{total_genes}_{iters}.npz"

    if os.path.isfile(dist_file):
        print("Distribution file exists. Now we have to read it.")
        return np.load(dist_file)["distribution"]

    print("Distribution file does not exist. Now we have to generate it.")

    # n = total_genes  # Sampling size for random distribution
    ranks = np.linspace(start=1, stop=total_genes, num=total_genes)
    ranks = (ranks - 0.5) / total_genes

    dist = Parallel(n_jobs=-1, verbose=5, backend="multiprocessing")(
        delayed(distribution_worker)(max_target, ranks) for _ in range(iters)
    )
    dist = np.array(dist).T
    np.savez_compressed(file=dist_file, distribution=dist)
    return dist


def sample_worker(
    sample: pd.DataFrame,
    prior_network: pd.DataFrame,
    distribution: np.array,
    iters: int,
):
    sample.dropna(inplace=True)
    sample["rank"] = sample.rank(ascending=False)
    sample["rank"] = (sample["rank"] - 0.5) / len(sample)
    sample["rev_rank"] = 1 - sample["rank"]

    # Get target genes rank
    for tf_id, tf_row in prior_network.iterrows():
        targets = tf_row["target"]
        actions = tf_row["action"]

        acti_rs = 0
        inhi_rs = 0
        valid_targets = 0

        for i, action in enumerate(actions):
            if targets[i] in sample.index:
                valid_targets += 1
                if action == 1:
                    acti_rs += np.average(sample.loc[targets[i], "rank"])
                    inhi_rs += np.average(sample.loc[targets[i], "rev_rank"])
                else:
                    inhi_rs += np.average(sample.loc[targets[i], "rank"])
                    acti_rs += np.average(sample.loc[targets[i], "rev_rank"])

        if acti_rs == 0 and inhi_rs == 0:  # No ranks for the all target genes
            prior_network.loc[tf_id, "rs"] = np.nan
            prior_network.loc[tf_id, "valid_target"] = np.nan
        else:
            if valid_targets >= 3:  # At least 3 valid target genes
                rs = np.min([acti_rs, inhi_rs])
                rs = rs / valid_targets  # Average rank-sum
                prior_network.loc[tf_id, "rs"] = rs if acti_rs < inhi_rs else -1 * rs
                prior_network.loc[tf_id, "valid_target"] = valid_targets
            else:  # Less than 3 valid target genes
                prior_network.loc[tf_id, "rs"] = np.nan
                prior_network.loc[tf_id, "valid_target"] = np.nan

    # Counting how many times the rs is less than the random distribution
    for idx1, row1 in prior_network.iterrows():
        # If rs is Nan, then skip
        if np.isnan(row1["rs"]):
            prior_network.loc[idx1, "count"] = np.nan
            prior_network.loc[idx1, "p-value"] = np.nan
            continue

        # n_dist = distribution[row1['updown'] - 1] # 1. Getting distribution of all targets
        n_dist = distribution[
            int(row1["valid_target"]) - 1
        ]  # 2. Getting distribution of valid targets only
        count = (
            np.sum(np.abs(n_dist) <= np.abs(row1["rs"])) + 1
        )  # Add 1 to avoid zero division

        if row1["rs"] < 0:
            count = -1 * count

        prior_network.loc[idx1, "count"] = count
        prior_network.loc[idx1, "p-value"] = count / iters

    return prior_network["p-value"].values

    # # Return as "p-values, (updown, valid_target)" format
    # prior_network["combined"] = (
    #     prior_network["p-value"].astype(str)
    #     + ", ("
    #     + prior_network["updown"].astype(str)
    #     + " "
    #     + "["
    #     + prior_network["up"].astype(str)
    #     + ","
    #     + prior_network["down"].astype(str)
    #     + "], "
    #     + round(prior_network["valid_target"]).astype(str)
    #     + ")"
    # )  # p-values, (updown, valid_target)
    #
    # return prior_network["combined"].values


def run_analysis(
    tfs,
    gene_exp: pd.DataFrame,
    prior_network: pd.DataFrame,
    distribution: np.array,
    iters: int,
):
    gene_exp = gene_exp.T
    parallel = Parallel(n_jobs=-1, verbose=5, backend="multiprocessing")
    output = parallel(
        delayed(sample_worker)(pd.DataFrame(row), prior_network, distribution, iters)
        for idx, row in gene_exp.iterrows()
    )
    output = pd.DataFrame(output, columns=tfs, index=gene_exp.index)
    return output


def main(prior_network: pd.DataFrame, gene_exp: pd.DataFrame, iters: int):
    # prior_network = pd.read_csv(prior_file, sep='\t')
    # gene_exp = pd.read_csv(gene_exp_file, sep='\t', index_col=0)

    gene_exp = gene_exp.apply(zscore, axis=1, nan_policy="omit")

    # Grouping prior_network network
    prior_network = prior_network.groupby("tf").agg({"action": list, "target": list})
    prior_network["updown"] = prior_network["target"].apply(lambda x: len(x))

    # prior_network["up"] = prior_network["action"].apply(lambda x: x.count(1.0))  # Added Later
    # prior_network["down"] = prior_network["action"].apply(lambda x: x.count(-1.0))  # Added Later

    distribution = get_distribution(
        max_target=np.max(prior_network["updown"]),
        total_genes=len(gene_exp),
        iters=iters,
    )

    return run_analysis(
        tfs=prior_network.index,
        gene_exp=gene_exp,
        prior_network=prior_network,
        distribution=distribution,
        iters=iters,
    )


def read_mouse_to_human_mapping_file():
    mth_file = "data/mouse_to_human.tsv"
    if not os.path.isfile(mth_file):
        print("Mouse to human mapping file does not exist. Let's download it.\n")
        file_url = "https://www.cs.umb.edu/~kisan/data/mouse_to_human.tsv"
        response = requests.get(file_url)

        if response.status_code == 200:
            with open(mth_file, "wb") as f:
                f.write(response.content)
            print("Mouse to human mapping file downloaded successfully.")
        else:
            print(
                "Mouse to human mapping file could not be downloaded.",
                response.status_code,
            )
            sys.exit(1)

    return pd.read_csv(mth_file, sep="\t")


def read_data(p_file: str, g_file: str):
    prior_network = pd.read_csv(
        p_file,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["tf", "action", "target"],
        converters={
            "action": lambda x: (
                1
                if x == "upregulates-expression"
                else -1 if x == "downregulates-expression" else None
            )
        },
    ).dropna()
    prior_network = prior_network.reset_index(drop=True)

    gene_exp = pd.read_csv(g_file, sep="\t", index_col=0)

    # Mouse to human Mapping process
    mouse_to_human = read_mouse_to_human_mapping_file()

    # Replace index with human gene IDs, filter, clean, and explode
    gene_exp = gene_exp.rename(
        index=dict(zip(mouse_to_human["Mouse"], mouse_to_human["Human"]))
    )
    gene_exp = gene_exp[gene_exp.index.str.match(r"^\[.*\]$")]
    gene_exp.index = gene_exp.index.str.strip("[]")
    gene_exp = (
        gene_exp.assign(index=gene_exp.index.str.split(","))
        .explode("index")
        .reset_index(drop=True)
    )
    gene_exp = gene_exp.set_index("index")

    gene_exp = gene_exp.replace(0.0, np.nan).dropna(
        thresh=int(len(gene_exp.columns) * 0.05)
    )

    # Get only first 10 columns
    gene_exp = gene_exp.iloc[:, :10]

    return prior_network, gene_exp


if __name__ == "__main__":
    prior_net, gene_e = read_data("data/causal_priors.txt", "data/5knormalized_mat.tsv")

    p_values = main(prior_net, gene_e, 100_000)

    # Remove columns with all NaN values
    p_values.dropna(axis=1, how="all", inplace=True)

    pValFile = "data/test_p2.tsv"
    p_values.to_csv(pValFile, sep="\t")
