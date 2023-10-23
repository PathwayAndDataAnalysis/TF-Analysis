import argparse
import os

import pandas as pd
import numpy as np
import sys
import requests
from scipy.stats import zscore
from joblib import Parallel, delayed


def read_mouse_to_human_mapping_file():
    mth_file = 'data/mouse_to_human.tsv'
    if not os.path.isfile(mth_file):
        print('Mouse to human mapping file does not exist. Let\'s download it.\n')
        file_url = 'https://www.cs.umb.edu/~kisan/data/mouse_to_human.tsv'
        response = requests.get(file_url)

        if response.status_code == 200:
            with open(mth_file, 'wb') as f:
                f.write(response.content)
            print('Mouse to human mapping file downloaded successfully.')
        else:
            print('Mouse to human mapping file could not be downloaded.', response.status_code)
            sys.exit(1)

    return pd.read_csv(mth_file, sep='\t')


def prepare_data(cp_file: str, sc_file: str, is_sim_data: int):
    prepared_sc_file = sc_file.split('.')[0] + '_prepared.tsv'
    prepared_cp_file = cp_file.split('.')[0] + '_prepared.tsv'

    if not os.path.isfile(prepared_sc_file) or not os.path.isfile(prepared_cp_file):
        print('Prepared files do not exist. Now we have to prepare it.')

        # Read the causal-priors file
        priors = pd.read_csv(cp_file, sep='\t', header=None, usecols=[0, 1, 2],
                             names=['symbol', 'action', 'targetSymbol'])
        priors = priors[priors['action'].isin(['upregulates-expression', 'downregulates-expression'])]
        priors['isUp'] = np.where(priors['action'] == 'upregulates-expression', 1, -1)
        priors.drop(['action'], axis=1, inplace=True)

        # Save the prepared causal-priors file
        priors.to_csv(prepared_cp_file, sep='\t', index=False)

        # Read the single cell file
        norm_sc = pd.read_csv(sc_file, sep='\t', header=0, index_col=0)

        # If the data is not simulated
        if is_sim_data == '0' or is_sim_data == 0:
            # Read the mouse to human mapping file
            mouse_to_human = read_mouse_to_human_mapping_file()
            # Remove rows of mouse_to_human if not in norm_sc
            mouse_to_human = mouse_to_human[mouse_to_human['Mouse'].isin(norm_sc.index)]
            # breakdown mouse_to_human Human column into multiple rows if there are multiple human genes
            mouse_to_human['Human'] = mouse_to_human['Human'].str.split(', ')
            mouse_to_human = mouse_to_human.explode('Human')
            # Remove [ and ] from mouse_to_human Human column
            mouse_to_human['Human'] = mouse_to_human['Human'].str.replace('[', '').str.replace(']', '')

            # Add into array symbol and targetSymbol of priors column
            priors_symbol = priors['symbol'].values
            priors_targetSymbol = priors['targetSymbol'].values

            # Merge priors_symbol and priors_targetSymbol into single array
            priors_symbol = np.concatenate((priors_symbol, priors_targetSymbol), axis=0)
            priors_symbol = np.unique(priors_symbol)
            mouse_to_human = mouse_to_human[mouse_to_human['Human'].isin(priors_symbol)]

            # Remove one to multiple mappings from mouse_to_human
            mouse_to_human = mouse_to_human[~mouse_to_human['Mouse'].duplicated(keep=False)]
            # Remove multiple to one mapping from mouse_to_human
            mouse_to_human = mouse_to_human[~mouse_to_human['Human'].duplicated(keep=False)]

            # Get the rows for each Mouse symbol from norm_sc and add to new dataframe
            sc_prepared = pd.DataFrame()
            for mouse_symbol in mouse_to_human['Mouse'].values:
                # sc_prepared = sc_prepared.append(norm_sc.loc[mouse_symbol])  # Does not support old pandas version
                sc_prepared = pd.concat([sc_prepared, norm_sc.loc[mouse_symbol].to_frame().T], axis=0)
            sc_prepared.index = mouse_to_human['Human'].values

            # Drop rows with all 0s
            sc_prepared = sc_prepared.loc[~(sc_prepared == 0).all(axis=1)]

            # Export sc_prepared to tsv
            sc_prepared.to_csv(prepared_sc_file, sep='\t')
            print('Prepared files saved successfully.')

        # If the data is simulated
        else:
            # Export norm_sc to tsv
            norm_sc.to_csv(prepared_sc_file, sep='\t')
            print('Prepared files saved successfully.')

    return prepared_cp_file, prepared_sc_file


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


def cell_worker(idx, row, cpo_df, distribution, iters, temp):
    cell = pd.DataFrame({'symbol': row.index, 'zscore': row.values})
    cell.dropna(inplace=True)
    cell.sort_values(by=['zscore'], ascending=[False], inplace=True)
    cell.reset_index(drop=True, inplace=True)

    cell['acti_rank'] = cell.index
    cell['acti_rank'] = (cell['acti_rank'] + 0.5) / len(cell)
    # Shuffle acti_rank
    # cell['acti_rank'] = np.random.choice(cell['acti_rank'], len(cell), replace=False)
    cell['inhi_rank'] = 1 - cell['acti_rank']

    cpo_df_c = cpo_df.copy()
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


def run_cell_worker(cpo_grouped_df, sc_df, cpo_df, distribution, iters):
    tfs = cpo_grouped_df['symbol'].values
    parallel = Parallel(n_jobs=-1, verbose=10, backend='multiprocessing')
    output = parallel(delayed(cell_worker)(idx, row, cpo_df, distribution, iters, tfs) for idx, row in sc_df.iterrows())
    output = pd.DataFrame(output, columns=tfs, index=sc_df.index)
    output.dropna(axis=1, how='all', inplace=True)
    return output


def main(cp_file: str, sc_file: str, iters: int, is_sim_data: int):
    # Print type of cp_file, sc_file, iters, is_sim_data
    print(type(cp_file), type(sc_file), type(iters), type(is_sim_data))
    prepared_cp_file, prepared_sc_file = prepare_data(cp_file, sc_file, is_sim_data)
    print(prepared_cp_file, prepared_sc_file, iters)

    # Read the prepared causal-priors file
    cpo_df = pd.read_csv(prepared_cp_file, sep='\t')

    # Read the prepared single cell file
    sc_df = pd.read_csv(prepared_sc_file, sep='\t', header=0, index_col=0).T
    sc_df.replace(0, np.nan, inplace=True)
    sc_df = pd.DataFrame(zscore(sc_df, nan_policy='omit'), index=sc_df.index, columns=sc_df.columns)

    # Remove rows of cpo_df if targetSymbol is not present in Symbols column of sc_df dataframe
    cpo_df = cpo_df[cpo_df['targetSymbol'].isin(sc_df.columns)]
    cpo_df = cpo_df.reset_index(drop=True)

    # Group cpo_df by symbol
    cpo_grouped_df = cpo_df.groupby('symbol')['isUp'].apply(list).reset_index(name='upDownList')
    cpo_grouped_df['targetList'] = cpo_df.groupby('symbol')['targetSymbol'].apply(list).reset_index(name='targetList')[
        'targetList']
    cpo_grouped_df['upDownCount'] = cpo_grouped_df['upDownList'].apply(lambda x: len(x))
    max_target = np.max(cpo_grouped_df['upDownCount'])

    # Read or Generate the distribution
    distribution = get_distribution(max_target, iters)

    return run_cell_worker(cpo_grouped_df, sc_df, cpo_df, distribution, iters)


if __name__ == "__main__":
    # Get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--causal-priors-file', help='Causal-priors file path', required=True)
    parser.add_argument('-sc', '--single-cell-file', help='Single cell gene expression file path', required=True)
    parser.add_argument('-iters', '--iterations', help='Number of iterations', required=True)
    parser.add_argument('-o', '--output-file', help='Output file path', required=True)
    parser.add_argument('-sim', '--simulated-data',
                        help='Is this simulated data, 0 for non-simulated data and 1 for simulated data',
                        required=False, default=0,
                        type=int)
    args = parser.parse_args()
    print(args)

    # Check if the files exist
    if not os.path.isfile(args.causal_priors_file):
        print('Causal-priors file does not exist')
        sys.exit(1)

    if not os.path.isfile(args.single_cell_file):
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
    p_values = main(args.causal_priors_file, args.single_cell_file, args.iterations, args.simulated_data)
    # pValFile = "data/p-values_" + str(args.iterations) + "real_data.tsv"
    pValFile = args.output_file
    p_values.to_csv(pValFile, sep='\t')
    print('p-values saved successfully.')
