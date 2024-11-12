import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
from joblib import Parallel, delayed


def distribution_worker(max_target: int, ranks: np.array):
    arr = np.zeros(max_target)
    cs = np.random.choice(ranks, max_target, replace=False).cumsum()
    for i in range(0, max_target):
        amr = cs[i] / (i + 1)
        arr[i] = np.min([amr, 1 - amr])
    return arr


def get_distribution(max_target: int, total_genes: int, iters: int):
    dist_file = 'data/distribution_' + str(max_target) + '_' + str(iters) + '.npz'

    if os.path.isfile(dist_file):
        print('Distribution file exists. Now we have to read it.')
        return np.load(dist_file)['distribution']

    print('Distribution file does not exist. Now we have to generate it.')

    # n = 10_000  # Sampling size for random distribution
    n = total_genes  # Sampling size for random distribution
    ranks = np.linspace(start=1, stop=total_genes, num=n)
    ranks = (ranks - 0.5) / total_genes

    dist = Parallel(n_jobs=-1, verbose=5, backend='multiprocessing')(
        delayed(distribution_worker)
        (max_target, ranks)
        for _ in range(iters)
    )
    dist = np.array(dist).T
    np.savez_compressed(file=dist_file, distribution=dist)
    return dist


def sample_worker(sample: pd.DataFrame, prior_network: pd.DataFrame, distribution: np.array, iters: int):
    sample['rank'] = sample.rank(ascending=False)
    sample['rank'] = (sample['rank'] - 0.5) / len(sample)
    sample['rev_rank'] = 1 - sample['rank']

    # Get target genes rank
    for tf_id, tf_row in prior_network.iterrows():
        targets = tf_row['target']
        actions = tf_row['action']

        acti_rs = 0
        inhi_rs = 0

        for i, action in enumerate(actions):
            if action == 1:
                acti_rs += sample.loc[targets[i], 'rank']
                inhi_rs += sample.loc[targets[i], 'rev_rank']
            else:
                inhi_rs += sample.loc[targets[i], 'rank']
                acti_rs += sample.loc[targets[i], 'rev_rank']

        rs = np.min([acti_rs, inhi_rs])
        rs = rs / len(targets)  # Average rank-sum

        prior_network.loc[tf_id, 'rs'] = rs if acti_rs < inhi_rs else -1 * rs

    # Counting how many times the rs is less than the random distribution
    for idx1, row1 in prior_network.iterrows():
        n_dist = distribution[row1['updown'] - 1]
        count = np.sum(np.abs(n_dist) <= np.abs(row1['rs'])) + 1  # Add 1 to avoid zero division

        if row1['rs'] < 0:
            count = -1 * count

        prior_network.loc[idx1, 'count'] = count
        prior_network.loc[idx1, 'p-value'] = count / iters

    return prior_network['p-value'].values


def run_analysis(tfs, gene_exp: pd.DataFrame, prior_network: pd.DataFrame, distribution: np.array, iters: int):
    gene_exp = gene_exp.T
    parallel = Parallel(n_jobs=-1, verbose=5, backend='multiprocessing')
    output = parallel(
        delayed(sample_worker)
        (pd.DataFrame(row), prior_network, distribution, iters)
        for idx, row in gene_exp.iterrows()
    )
    output = pd.DataFrame(output, columns=tfs, index=gene_exp.index)
    return output


def main(prior_file: str, gene_exp_file: str, iters: int):
    prior_network = pd.read_csv(prior_file, sep='\t')
    gene_exp = pd.read_csv(gene_exp_file, sep='\t', index_col=0)

    gene_exp = gene_exp.apply(zscore, axis=1)

    # Grouping prior_network network
    prior_network = prior_network.groupby('tf').agg({'action': list, 'target': list})
    prior_network['updown'] = prior_network['target'].apply(lambda x: len(x))

    distribution = get_distribution(max_target=np.max(prior_network['updown']),
                                    total_genes=len(gene_exp),
                                    iters=iters)

    return run_analysis(tfs=prior_network.index,
                        gene_exp=gene_exp,
                        prior_network=prior_network,
                        distribution=distribution,
                        iters=iters)


if __name__ == "__main__":
    g_file = "data/demo3_data.tsv"
    p_file = "data/demo3_priors.tsv"

    p_values = main(p_file, g_file, 100_000)
    pValFile = "data/demo3_pvals.tsv"
    p_values.to_csv(pValFile, sep='\t')
