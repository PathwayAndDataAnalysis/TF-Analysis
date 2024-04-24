import argparse
import os

import pandas as pd
import numpy as np
import sys
from scipy.stats import zscore
from joblib import Parallel, delayed


def distribution_worker(iter, max_target, ranks):
    arr = np.zeros(max_target)
    cs = np.random.choice(ranks, max_target, replace=False).cumsum()
    for target in range(1, max_target + 1):
        amr = cs[target - 1] / target
        arr[target - 1] = np.min([amr, 1 - amr])
    return arr


def get_distribution(max_target: int, iters: int):
    dist_file = 'data/distribution_' + str(max_target) + '_' + str(iters) + '.npz'

    if os.path.isfile(dist_file):
        print('Distribution file exists. Now we have to read it.')
        return np.load(dist_file)['distribution']

    print('Distribution file does not exist. Now we have to generate it.')
    n = 10_000
    ranks = [(i + 0.5) / n for i in range(0, n)]

    dist = Parallel(n_jobs=-1, verbose=5, backend='multiprocessing')(
        delayed(distribution_worker)(itr, max_target, ranks) for itr in range(iters))
    dist = np.array(dist).T
    dist = np.sort(dist, axis=1)
    np.savez_compressed(dist_file, distribution=dist)
    return dist


def cell_worker(idx, row, prior_network, distribution, iters, temp):
    cell = pd.DataFrame({'symbol': row.index, 'zscore': row.values})
    cell.dropna(inplace=True)
    cell.sort_values(by=['zscore'], ascending=[False], inplace=True)
    cell.reset_index(drop=True, inplace=True)

    cell['acti_rank'] = cell.index
    cell['acti_rank'] = (cell['acti_rank'] + 0.5) / len(cell)
    # Shuffle acti_rank
    # cell['acti_rank'] = np.random.choice(cell['acti_rank'], len(cell), replace=False)
    cell['inhi_rank'] = 1 - cell['acti_rank']

    cpo_df_c = prior_network.copy()
    cpo_df_c = cpo_df_c[cpo_df_c['targetSymbol'].isin(cell['symbol'])]
    cpo_df_c.reset_index(drop=True, inplace=True)

    cpo_df_c['pos_rs'] = cpo_df_c.apply(
        lambda x:
        cell.loc[cell['symbol'] == x['targetSymbol'], 'acti_rank'].values[0]
        if x['isUp'] == 1
        else
        cell.loc[cell['symbol'] == x['targetSymbol'], 'inhi_rank'].values[0],
        axis=1
    )
    cpo_df_c['neg_rs'] = 1 - cpo_df_c['pos_rs']

    # Group
    cpo_df_c_grouped = cpo_df_c.groupby('symbol')['isUp'].apply(list).reset_index(name='upDownList')
    cpo_df_c_grouped['targetList'] = \
        cpo_df_c.groupby('symbol')['targetSymbol'].apply(list).reset_index(name='targetList')['targetList']
    cpo_df_c_grouped['upDownCount'] = cpo_df_c_grouped['upDownList'].apply(lambda x: len(x))
    cpo_df_c_grouped['pos_rs'] = cpo_df_c.groupby('symbol')['pos_rs'].apply(list).reset_index(name='pos_rs')[
        'pos_rs']
    cpo_df_c_grouped['neg_rs'] = cpo_df_c.groupby('symbol')['neg_rs'].apply(list).reset_index(name='neg_rs')[
        'neg_rs']
    cpo_df_c_grouped['rs'] = cpo_df_c_grouped.apply(
        lambda x: np.min([np.mean(x['pos_rs']), np.mean(x['neg_rs'])]), axis=1
    )
    cpo_df_c_grouped['rs'] = cpo_df_c_grouped.apply(
        lambda x: x['rs'] if np.mean(x['pos_rs']) > np.mean(x['neg_rs']) else -1 * x['rs'], axis=1
    )

    # Counting how many times the RS is less than the random distribution
    for idx1, row1 in cpo_df_c_grouped.iterrows():
        arr = distribution[row1['upDownCount'] - 1]
        rs_count = np.searchsorted(arr, np.abs(row1['rs']), side='right')
        if rs_count == 0:
            rs_count = 1
        if np.sign(row1['rs']) == -1:
            rs_count = -1 * rs_count
        cpo_df_c_grouped.loc[idx1, 'count'] = rs_count
        cpo_df_c_grouped.loc[idx1, 'p-value'] = rs_count / iters

    # get column symbol and p-value from cpo_df_c_grouped dataframe and create a new dataframe p_df
    p_df = pd.DataFrame({'symbol': cpo_df_c_grouped['symbol'], 'p-value': cpo_df_c_grouped['p-value']})

    temp_df = pd.DataFrame({'symbol': temp})
    temp_df = pd.merge(temp_df, p_df, on='symbol', how='left')

    return temp_df['p-value'].values


def run_cell_worker(tfs, gene_expr, prior_network, distribution, iters):
    parallel = Parallel(n_jobs=-1, verbose=10, backend='multiprocessing')
    output = parallel(
        delayed(cell_worker)(idx, row, prior_network, distribution, iters, tfs) for idx, row in gene_expr.iterrows())
    output = pd.DataFrame(output, columns=tfs, index=gene_expr.index)
    output.dropna(axis=1, how='all', inplace=True)
    return output


def main(cp_file: str, sc_file: str, iters: int):
    # Read the prepared causal-priors file
    prior_network = pd.read_csv(cp_file, sep='\t')

    # Read the prepared single cell file
    gene_expr = pd.read_csv(sc_file, sep='\t', header=0, index_col=0).T
    gene_expr.replace(0, np.nan, inplace=True)

    # Normalize the gene expression data using z-score
    gene_expr = pd.DataFrame(zscore(gene_expr, nan_policy='omit'), index=gene_expr.index, columns=gene_expr.columns)

    # Remove rows of prior_network if targetSymbol is not present in Symbols column of gene_expr dataframe
    prior_network = prior_network[prior_network['targetSymbol'].isin(gene_expr.columns)]
    prior_network = prior_network.reset_index(drop=True)

    # Group prior_network by symbol
    prior_net_grouped = prior_network.groupby('symbol')['isUp'].apply(list).reset_index(name='upDownList')
    prior_net_grouped['targetList'] = prior_network.groupby(
        'symbol')['targetSymbol'].apply(list).reset_index(name='targetList')['targetList']
    prior_net_grouped['upDownCount'] = prior_net_grouped['upDownList'].apply(lambda x: len(x))
    max_target = np.max(prior_net_grouped['upDownCount'])

    # Read or Generate the distribution
    distribution = get_distribution(max_target, iters)

    return run_cell_worker(prior_net_grouped['symbol'].values, gene_expr, prior_network, distribution, iters)


if __name__ == "__main__":
    # Get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-prior-net', '--prior-network-file', help='Causal-priors file path', required=True)
    parser.add_argument('-gene-exp', '--gene-expression-file', help='Single cell gene expression file path',
                        required=True)
    parser.add_argument('-iters', '--iterations', help='Number of iterations', required=True)
    parser.add_argument('-output', '--output-file', help='Output file path', required=True)

    args = parser.parse_args()
    print(args)

    # Check if the files exist
    if not os.path.isfile(args.prior_network_file):
        print('Prior network file does not exist')
        sys.exit(1)

    if not os.path.isfile(args.gene_expression_file):
        print('Single cell gene expression file does not exist')
        sys.exit(1)

    # Check if the number of iterations is an integer
    try:
        args.iterations = int(args.iterations)
    except ValueError:
        print('Number of iterations should be an integer')
        sys.exit(1)

    # Check if the number of iterations is greater than 0
    if args.iterations <= 0:
        print('Number of iterations should be greater than 0')
        sys.exit(1)

    # Run the main function
    p_values = main(args.prior_network_file, args.gene_expression_file, args.iterations)
    pValFile = args.output_file
    p_values.to_csv(pValFile, sep='\t')

    # Get absolute values of p-values
    p_values = p_values.abs()
    p_values.to_csv("results_abs.tsv", sep='\t')
    print('p-values saved successfully.')
