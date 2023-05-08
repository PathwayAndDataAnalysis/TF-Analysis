import sys

import numpy as np
import pandas as pd
from timeit import default_timer as timer


def get_rank_sum(network: pd.DataFrame, rank_df: pd.DataFrame):
    """
    :param network: Causal-priors network in pd.DataFrame format
    :param rank_df:  Differential expression data in pd.DataFrame format.
    Also contains rank and reverse_rank columns
    :return: Actual rank-sum of genes in de dataframe
    """
    # find the maximum number of targets for a Symbol
    max_targets = max(network['Symbols'].value_counts().to_dict().values())

    # Create a column named is_upregulated in network dataframe, 1 is upregulated and -1 is downregulated
    network['is_upregulated'] = np.where(network['action'] == 'upregulates-expression', 1, -1)
    # Delete action column from network dataframe
    network.drop('action', axis=1, inplace=True)

    # Find the positive and negative ranks of targetSymbols
    network['rank'] = network['targetSymbol'].apply(lambda x: rank_df[rank_df['Symbols'] == x]['rank'].values[0])
    network['reverse_rank'] = network['targetSymbol'].apply(
        lambda x: rank_df[rank_df['Symbols'] == x]['reverse_rank'].values[0])

    # Calculate the actual rank sum of each Symbol for positive and negative ranks
    actual_rank_sum_df = network.groupby('Symbols').apply(
        lambda x: x[x['is_upregulated'] == 1]['rank'].sum() + x[x['is_upregulated'] == -1][
            'reverse_rank'].sum()).reset_index(name='actual_rank_sum')
    actual_rank_sum_df['actual_negative_rank_sum'] = network.groupby('Symbols').apply(
        lambda x: x[x['is_upregulated'] == 1]['reverse_rank'].sum() + x[x['is_upregulated'] == -1][
            'rank'].sum()).reset_index(name='actual_negative_rank_sum')['actual_negative_rank_sum']

    # Create a new dataframe with key: Symbols and value: List of 1's and -1's
    target_counts_df = network.groupby('Symbols')['is_upregulated'].apply(list).reset_index(name='count')

    # Merge actual_rank_sum_df dataframe with target_counts_df dataframe on Symbols column
    target_counts_df = pd.merge(target_counts_df, actual_rank_sum_df, on='Symbols')
    # Clear actual_rank_sum_df dataframe from memory
    del actual_rank_sum_df

    # Choose minimum between actual_rank_sum and actual_negative_rank_sum as actual_min_rank_sum
    target_counts_df['actual_min_rank_sum'] = target_counts_df.apply(
        lambda x: min(x['actual_rank_sum'], x['actual_negative_rank_sum']), axis=1)
    # Find which one is minimum and store it in IsPosNeg column
    target_counts_df['IsPosNeg'] = target_counts_df.apply(
        lambda x: 'Positive' if x['actual_rank_sum'] < x['actual_negative_rank_sum'] else 'Negative',
        axis=1)
    # Drop actual_rank_sum and actual_negative_rank_sum columns from target_counts_df dataframe
    target_counts_df.drop(['actual_rank_sum', 'actual_negative_rank_sum'], axis=1, inplace=True)

    # Remove the rows where length of count is less than 3
    target_counts_df = target_counts_df[target_counts_df['count'].apply(lambda x: len(x)) >= 3]

    # Count the size of count column and store it in count_size column
    target_counts_df['totalCount'] = target_counts_df['count'].apply(lambda x: len(x))

    # Find the max rank from rank_df dataframe
    max_rank = rank_df['rank'].max()

    # Add a new column in target_counts_df to store how many times the rank sum is less than the actual rank sum
    target_counts_df['rank_sum_less_than_actual'] = 0

    unique_total_count = target_counts_df['totalCount'].unique()

    rand_iter = 1_000

    # Initialize the array to store results
    results_array = np.zeros((len(unique_total_count), rand_iter))

    # Check how many times this for loop takes to run
    loop_start = timer()
    for i in range(rand_iter):
        # Draw a random number from 0 to max_rank+1
        drawn_list = np.random.choice(max_rank + 1, max_targets, replace=False)
        reverse_drawn_list = max_rank - drawn_list
        min_rank_sum = np.minimum(np.cumsum(drawn_list), np.cumsum(reverse_drawn_list))
        results_array[:, i] = [min_rank_sum[x-1] for x in unique_total_count]

    loop_end = timer()
    print(f'Loop took {loop_end - loop_start} seconds')  # took 28.592077041015727 seconds

    # Merge all columns of results_array into a list of values
    results_array = pd.DataFrame(results_array).apply(lambda x: x.values.tolist(), axis=1)
    df_result = pd.concat([pd.DataFrame(unique_total_count), results_array], axis=1)
    df_result.columns = ['totalCount', 'rank_sum_list']

    # Merge the rank_sum_list column with target_counts_df dataframe
    target_counts_df['rank_sum_list'] = target_counts_df['totalCount'].apply(
        lambda x: df_result[df_result['totalCount'] == x]['rank_sum_list'].values[0])

    # Count the numbers in rank_sum_list which are less than actual_min_rank_sum and store it in a new column
    target_counts_df['rank_sum_less_than_actual'] = target_counts_df.apply(
        lambda x: len([i for i in x['rank_sum_list'] if i < x['actual_min_rank_sum']]), axis=1)

    # Calculate the p-value
    target_counts_df['p-value'] = target_counts_df['rank_sum_less_than_actual'].apply(
        lambda x: (x + 1) / rand_iter if x == 0 else x / rand_iter)

    # Drop the columns which are not required
    target_counts_df.drop(['rank_sum_list'], axis=1, inplace=True)

    # Save the dataframe to a csv file
    target_counts_df.to_csv('../output/final_output_file.csv', index=False)


def main(cp_file: str, de_file: str):
    """
    :param cp_file: File path of causal-priors file
    :param de_file: File path of differential-exp file
    :return:
    """
    if cp_file is None or de_file is None:
        print('Please provide causal-priors and differential-exp file path')
        sys.exit(1)

    if not isinstance(cp_file, str) or not isinstance(de_file, str):
        print('Please provide causal-priors and differential-exp file path as string')
        sys.exit(1)

    try:
        cp = pd.read_csv(cp_file, sep='\t', header=None)
        cp.columns = ['Symbols', 'action', 'targetSymbol', 'reference', 'residual']
        cp = cp.drop(['reference', 'residual'], axis=1)
        cp = cp[cp['action'].isin(['upregulates-expression', 'downregulates-expression'])]
        cp = cp.reset_index(drop=True)

        de = pd.read_csv(de_file, sep='\t')

        # Create a new column named updown based on positive
        # and negative values of SignedP column of de dataframe
        de['updown'] = np.where(de['SignedP'] > 0, '1', '-1')

        # Sort SignedP column in ascending order if updown column is 1
        # and sort absolute values of SignedP column in ascending order if updown column is -1
        de = de.sort_values(by=['updown', 'SignedP'], ascending=[False, True])

        # Remove rows of cp dataframe if targetSymbol is not present in Symbols column of rank_df dataframe
        cp = cp[cp['targetSymbol'].isin(de['Symbols'])]
        # Reset index
        cp = cp.reset_index(drop=True)

        # Add new column named rank to de dataframe
        de['rank'] = np.arange(len(de))

        # Add reverse_rank column to de dataframe
        de['reverse_rank'] = de['rank'].max() - de['rank']

        # Sort Symbols column in ascending order of cp dataframe
        cp = cp.sort_values(by=['Symbols'], ascending=True, ignore_index=True)

        get_rank_sum(cp, de)

    except FileNotFoundError:
        print('File not found: ', cp_file)
        sys.exit(1)

    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == '__main__':
    priors_file = '../../data/causal-priors.txt'
    diff_file = '../../data/differential-exp.tsv'

    start = timer()
    main(priors_file, diff_file)
    end = timer()
    print("Time taken: ", end - start)
