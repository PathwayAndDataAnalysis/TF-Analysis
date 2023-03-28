import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_actual_rank_sum(network: pd.DataFrame, rank_df: pd.DataFrame):
    """
    :param network: Causal-priors network in pd.DataFrame format
    :param rank_df:  Differential expression data in pd.DataFrame format.
    Also contains rank and reverse_rank columns
    :return: Actual rank-sum of genes in de dataframe
    """

    # Create a new column named rank in cp dataframe
    network['rank'] = 0

    # Create a new column named reverse_rank in cp dataframe
    network['reverse_rank'] = 0

    # Find the rank of targetSymbol in Symbols column of rank_df dataframe
    # and assign it to rank column of network dataframe
    for i in range(len(network)):
        network.loc[i, 'rank'] = rank_df[rank_df['Symbols'] == network.loc[i, 'targetSymbol']]['rank'].values[0]
        network.loc[i, 'reverse_rank'] = \
            rank_df[rank_df['Symbols'] == network.loc[i, 'targetSymbol']]['reverse_rank'].values[0]
        network.loc[i, 'SignedP'] = rank_df[rank_df['Symbols'] == network.loc[i, 'targetSymbol']]['SignedP'].values[0]

    # Create a column named is_upregulated in network dataframe
    # Insert 1 if action column is upregulates-expression
    # Insert -1 if action column is downregulates-expression
    network['is_upregulated'] = np.where(network['action'] == 'upregulates-expression', 1, -1)

    # Calculate Rank-Sum
    # There are two types of rank-sum
    # 1. Positive View: For rows where updown column is 1, add rank column values of network dataframe
    #                   and for rows where updown column is -1, add reverse_rank column values of network dataframe
    # 2. Negative View: For rows where updown column is 1, add reverse_rank column values of network dataframe
    #                   and for rows where updown column is -1, add rank column values of network dataframe

    # Positive View
    positive_view = network[network['is_upregulated'] == 1]['rank'].sum() + \
                    network[network['is_upregulated'] == -1]['reverse_rank'].sum()
    # Negative View
    negative_view = network[network['is_upregulated'] == 1]['reverse_rank'].sum() + \
                    network[network['is_upregulated'] == -1]['rank'].sum()

    # print('Positive View: ', positive_view)
    # print('Negative View: ', negative_view)

    return positive_view, negative_view


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

    # Remove rows of rank_df dataframe if Symbols is not present in targetSymbol column of cp dataframe
    de = de[de['Symbols'].isin(cp['targetSymbol'])]
    # Reset index
    de = de.reset_index(drop=True)

    # Add new column named rank to de dataframe
    de['rank'] = np.arange(len(de))

    # Add reverse_rank column to de dataframe
    de['reverse_rank'] = de['rank'].max() - de['rank']

    # Sort Symbols column in ascending order of cp dataframe
    cp = cp.sort_values(by=['Symbols'], ascending=True, ignore_index=True)

    actual_ptive_rank_sum, actual_ntive_rank_sum = get_actual_rank_sum(cp, de)

    print('Actual Positive Rank Sum: ', actual_ptive_rank_sum)
    print('Actual Negative Rank Sum: ', actual_ntive_rank_sum)


    # Randomize rank column of de dataframe
    rank_sums = []
    for i in range(1000):
        # Randomize rank column of de dataframe
        de['rank'] = np.random.permutation(de['rank'])

        # Add reverse_rank column to de dataframe
        de['reverse_rank'] = de['rank'].max() - de['rank']

        positive_view, negative_view = get_actual_rank_sum(cp, de)

        # append smaller value
        if positive_view < negative_view:
            rank_sums.append(positive_view)
        else:
            rank_sums.append(negative_view)

    # Plot histogram of rank_sums
    plt.hist(rank_sums, density=True, bins=20, edgecolor='black')
    plt.xlabel('Rank Sum')
    plt.ylabel('Probability')
    plt.title('Histogram of Rank Sum')
    plt.savefig('hist.png')
    plt.show()

    print(cp.head())
    print(de.head())


if __name__ == '__main__':
    priors_file = '../data/causal-priors.txt'  # sys.argv[1]
    diff_file = '../data/differential-exp.tsv'  # sys.argv[2]

    main(priors_file, diff_file)
