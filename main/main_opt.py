import random
import sys

import numpy as np
import pandas as pd


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
    # Drop actual_rank_sum and actual_negative_rank_sum columns from target_counts_df dataframe
    target_counts_df.drop(['actual_rank_sum', 'actual_negative_rank_sum'], axis=1, inplace=True)

    # Remove the rows where length of count is less than 3
    target_counts_df = target_counts_df[target_counts_df['count'].apply(lambda x: len(x)) >= 3]

    # Create a column named up_down_tuple in target_counts_df df, (x, y) x count of 1's and y count of -1's
    target_counts_df['up_down_tuple'] = target_counts_df['count'].apply(lambda x: (x.count(1), x.count(-1)))

    # Find the max rank from rank_df dataframe
    max_rank = rank_df['rank'].max()

    # Add a new column in target_counts_df to store how many times the rank sum is less than the actual rank sum
    target_counts_df['rank_sum_less_than_actual'] = 0

    # Find the unique values in up_down_tuple column and store it in a pandas dataframe
    updown_df = pd.DataFrame(target_counts_df['up_down_tuple'].unique(), columns=['up_down_tuple'])

    rand_iter = 1_000_000
    for i in range(rand_iter):
        # Pick max_targets random numbers from 0 to max_rank+1
        randomly_drawn_list = np.random.randint(0, max_rank + 1, max_targets)

        # Create reverse randomly_drawn_list from rank_df dataframe
        reverse_randomly_drawn_list = max_rank - np.array(randomly_drawn_list)

        # Create a new df
        # df = pd.DataFrame(columns=[i])
        # updown_df['up_down_tuple'].apply(
        #     lambda x: sum(randomly_drawn_list[:x[0]]) + sum(reverse_randomly_drawn_list[x[0]:x[0]+x[1]]))
        # updown_df['up_down_tuple'].apply(lambda x: randomly_drawn_list[:x[0]])

        # Concatenate updown_df and df and store it in updown_df
        # updown_df = pd.concat((updown_df, df), axis=1)

    # Drop unnecessary columns from target_counts_df dataframe
    target_counts_df.drop(['count', 'up_down_tuple', 'random_ranks', 'negative_random_ranks',
                           'rank_sum', 'actual_min_rank_sum', 'negative_rank_sum'], axis=1, inplace=True)

    # Add column p-value by dividing rank_sum_less_than_actual by rand_iter
    # if rank_sum_less_than_actual is less than zero, then add 1 to it and then divide it by rand_iter
    target_counts_df['p-value'] = target_counts_df['rank_sum_less_than_actual'].apply(
        lambda x: (x + 1) / rand_iter if x == 0 else x / rand_iter)

    # Add another column is_min_enough by checking if auc is less than 0.15
    target_counts_df['is_min_enough'] = target_counts_df['p-value'] < 0.15

    # Save the dataframe to a csv file
    target_counts_df.to_csv('../data/output_file100k.csv', index=False)


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

    # Try except block to handle file not found error
    try:
        cp = pd.read_csv(cp_file, sep='\t', header=None)
        cp.columns = ['Symbols', 'action', 'targetSymbol', 'reference', 'residual']

        # remove all columns except upregulates-expression and downregulates-expression
        cp = cp[cp['action'].isin(['upregulates-expression', 'downregulates-expression'])]
        # reset index
        cp = cp.reset_index(drop=True)

        # delete reference and residual columns
        cp = cp.drop(['reference', 'residual'], axis=1)

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
    priors_file = '../data/causal-priors.txt'
    diff_file = '../data/differential-exp.tsv'

    main(priors_file, diff_file)
