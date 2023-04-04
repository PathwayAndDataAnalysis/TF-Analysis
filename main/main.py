import random
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def get_rank_sum(network: pd.DataFrame, rank_df: pd.DataFrame):
    """
    :param network: Causal-priors network in pd.DataFrame format
    :param rank_df:  Differential expression data in pd.DataFrame format.
    Also contains rank and reverse_rank columns
    :return: Actual rank-sum of genes in de dataframe
    """
    # Find how many Symbols are repeated in network dataframe and store it in a dictionary
    # key: Symbol
    # value: Number of times Symbol is repeated
    symbol_count = network['Symbols'].value_counts().to_dict()

    # Create a column named is_upregulated in network dataframe
    # Insert 1 if action column is upregulates-expression
    # Insert -1 if action column is downregulates-expression
    network['is_upregulated'] = np.where(network['action'] == 'upregulates-expression', 1, -1)

    # Find the ranks of targetSymbols in Symbols column of rank_df dataframe and add it to a new column named rank
    # in network dataframe
    network['rank'] = network['targetSymbol'].apply(lambda x: rank_df[rank_df['Symbols'] == x]['rank'].values[0])
    network['reverse_rank'] = network['targetSymbol'].apply(
        lambda x: rank_df[rank_df['Symbols'] == x]['reverse_rank'].values[0])

    # Group by Symbols column also add rank value if is_upregulated column is 1 and add reverse_rank value if
    # is_upregulated column is -1 to group
    # and store it in a dataframe
    # key: Symbols
    # value: Sum of ranks if is_upregulated column is 1 and sum of reverse_ranks if is_upregulated column is -1
    actual_rank_sum_df = network.groupby('Symbols').apply(
        lambda x: x[x['is_upregulated'] == 1]['rank'].sum() + x[x['is_upregulated'] == -1][
            'reverse_rank'].sum()).reset_index(name='actual_rank_sum')

    # Add new column named negative_rank_sum in actual_rank_sum_df dataframe. If is_upregulated column is 1
    # then add reverse_rank value to negative_rank_sum column and if is_upregulated column is -1 then add rank value
    # to negative_rank_sum column
    actual_rank_sum_df['actual_negative_rank_sum'] = network.groupby('Symbols').apply(
        lambda x: x[x['is_upregulated'] == 1]['reverse_rank'].sum() + x[x['is_upregulated'] == -1][
            'rank'].sum()).reset_index(name='actual_negative_rank_sum')['actual_negative_rank_sum']

    # Count how many 1's and -1's are there in is_upregulated column for each Symbol
    # and store it in a dataframe
    # key: Symbols
    # value: List of 1's and -1's
    target_counts_df = network.groupby('Symbols')['is_upregulated'].apply(list).reset_index(name='count')

    # Merge actual_rank_sum_df dataframe with target_counts_df dataframe on Symbols column
    target_counts_df = pd.merge(target_counts_df, actual_rank_sum_df, on='Symbols')
    # Clear actual_rank_sum_df dataframe from memory
    # del actual_rank_sum_df

    # Create a column named actual_min_rank_sum in target_counts_df dataframe, which is the minimum of
    # actual_rank_sum and actual_negative_rank_sum
    target_counts_df['actual_min_rank_sum'] = target_counts_df.apply(
        lambda x: min(x['actual_rank_sum'], x['actual_negative_rank_sum']), axis=1)

    # Create another column by counting how many 1's and -1's are there in count column and create a
    # tuple(count of 1's, count of -1's)
    # and store it in a dataframe
    # key: Symbols
    # value: Number of 1's and -1's
    target_counts_df['up_down_tuple'] = target_counts_df['count'].apply(lambda x: (x.count(1), x.count(-1)))

    # Remove rows where the sum of values in up_down_tuple is less than 3
    target_counts_df = target_counts_df[target_counts_df['up_down_tuple'].apply(lambda x: x[0] + x[1]) >= 3]

    # find the maximum number of times a Symbol is repeated
    max_targets = max(symbol_count.values())

    # Find the max rank from rank_df dataframe
    max_rank = rank_df['rank'].max()

    tp53_min_rank_sum_list = []
    tp53_rank_sum_list = []
    stat3_rank_sum_list = []
    jun_rank_sum_list = []
    irf1_rank_sum_list = []

    # Add a new column in target_counts_df to store how many times the rank sum is less than the actual rank sum
    target_counts_df['rank_sum_less_than_actual'] = 0

    rand_iter = 1_000
    for i in range(rand_iter):
        # Find the max_targets random numbers from 0 to max_rank
        # and store it in a list
        randomly_drawn_list = np.random.randint(0, max_rank + 1, max_targets)

        # Create reverse randomly_drawn_list from rank_df dataframe
        reverse_randomly_drawn_list = max_rank - np.array(randomly_drawn_list)

        # up_down_tuple is in the form of (x, y) where x is the number of 1's and y is the number of -1's
        # Choose x random numbers from randomly_drawn_list and y random numbers from reverse_randomly_drawn_list
        # and store it in a dataframe
        # key: Symbols
        # value: List of random numbers
        target_counts_df['random_ranks'] = target_counts_df['up_down_tuple'].apply(
            lambda x: random.sample(list(randomly_drawn_list), x[0]) + random.sample(list(reverse_randomly_drawn_list),
                                                                                     x[1]))

        # Find reverse rank from random_ranks column and store it in a new column named negative_random_ranks
        target_counts_df['negative_random_ranks'] = target_counts_df['random_ranks'].apply(
            lambda x: max_rank - np.array(x))

        # Add values in random_ranks and store it in a new column named rank_sum
        target_counts_df['rank_sum'] = target_counts_df['random_ranks'].apply(lambda x: sum(x))
        target_counts_df['negative_rank_sum'] = target_counts_df['negative_random_ranks'].apply(lambda x: sum(x))

        # Count how many times the rank_sum is less than the actual rank sum and s
        # tore it in a new column named rank_sum_less_than_actual
        target_counts_df['rank_sum_less_than_actual'] += target_counts_df.apply(
            lambda x: 1 if x['rank_sum'] < x['actual_min_rank_sum'] else 0, axis=1)

        # Add column p-value by dividing rank_sum_less_than_actual by rand_iter
        # if rank_sum_less_than_actual is less than zero, then add 1 to it and then divide it by rand_iter
        target_counts_df['p-value'] = target_counts_df['rank_sum_less_than_actual'].apply(
            lambda x: (x + 1) / rand_iter if x == 0 else x / rand_iter)

        # Add another column is_min_enough by checking if auc is less than 0.15
        target_counts_df['is_min_enough'] = target_counts_df['p-value'] < 0.15

        # Save the dataframe to a csv file
        target_counts_df.to_csv('../data/output_file100k.csv', index=False)

        # Collect the rank_sum values for TP53 value in Symbols column and store it in a list
        tp53_rank_sum_list.append(target_counts_df[target_counts_df['Symbols'] == 'TP53']['rank_sum'].values[0])
        # stat3_rank_sum_list.append(target_counts_df[target_counts_df['Symbols'] == 'STAT3']['rank_sum'].values[0])
        # jun_rank_sum_list.append(target_counts_df[target_counts_df['Symbols'] == 'JUN']['rank_sum'].values[0])
        # irf1_rank_sum_list.append(target_counts_df[target_counts_df['Symbols'] == 'MYC']['rank_sum'].values[0])

        # Collect the value which is minimum between rank_sum and negative_rank_sum values for TP53 value in Symbols
        # column and store it in a list
        tp53_min_rank_sum_list.append(min(target_counts_df[target_counts_df['Symbols'] == 'TP53']['rank_sum'].values[0],
                                          target_counts_df[target_counts_df['Symbols'] == 'TP53'][
                                              'negative_rank_sum'].values[0]))
        stat3_rank_sum_list.append(min(target_counts_df[target_counts_df['Symbols'] == 'STAT3']['rank_sum'].values[0],
                                       target_counts_df[target_counts_df['Symbols'] == 'STAT3'][
                                           'negative_rank_sum'].values[0]))
        jun_rank_sum_list.append(min(target_counts_df[target_counts_df['Symbols'] == 'JUN']['rank_sum'].values[0],
                                     target_counts_df[target_counts_df['Symbols'] == 'JUN']['negative_rank_sum'].values[
                                         0]))
        irf1_rank_sum_list.append(min(target_counts_df[target_counts_df['Symbols'] == 'IRF1']['rank_sum'].values[0],
                                      target_counts_df[target_counts_df['Symbols'] == 'IRF1'][
                                          'negative_rank_sum'].values[
                                          0]))


    # Plot tp53_min_rank_sum_list in histogram
    plt.hist(tp53_min_rank_sum_list, bins=100, edgecolor='white', color='#607c8e')
    plt.title('Histogram of min rank-sum for TP53')
    plt.xlabel('min Rank-sum')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    # actual_rank_sum = actual_rank_sum_df[actual_rank_sum_df['Symbols'] == 'TP53']['actual_rank_sum'].values[0]
    # Find the value which is minimum between rank_sum and negative_rank_sum values for TP53
    actual_rank_sum = min(actual_rank_sum_df[actual_rank_sum_df['Symbols'] == 'TP53']['actual_rank_sum'].values[0],
                          actual_rank_sum_df[actual_rank_sum_df['Symbols'] == 'TP53'][
                              'actual_negative_rank_sum'].values[0])
    # Plot a vertical line at actual rank_sum value for TP53
    plt.axvline(x=actual_rank_sum, color='r', linestyle='--')
    # save the plot
    plt.savefig('../data/tp53_min_rank_sum.png')
    plt.show()

    # Plot tp53_rank_sum_list in histogram
    plt.hist(tp53_rank_sum_list, bins=100, edgecolor='white', color='#607c8e')
    plt.title('Histogram of rank-sum for TP53')
    plt.xlabel('Rank-sum')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    # Find actual rank_sum value for TP53
    actual_rank_sum = actual_rank_sum_df[actual_rank_sum_df['Symbols'] == 'TP53']['actual_rank_sum'].values[0]
    # Plot a vertical line at actual rank_sum value for TP53
    plt.axvline(x=actual_rank_sum, color='r', linestyle='--')
    # save the plot
    plt.savefig('../data/tp53_rank_sum.png')
    plt.show()


    # Plot stat3_rank_sum_list in histogram
    plt.hist(stat3_rank_sum_list, bins=100, edgecolor='white', color='#607c8e')
    plt.title('Histogram of rank-sum for STAT3')
    plt.xlabel('Rank-sum')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    # Find actual rank_sum value for STAT3
    # actual_rank_sum = actual_rank_sum_df[actual_rank_sum_df['Symbols'] == 'STAT3']['actual_rank_sum'].values[0]
    # Find the value which is minimum between rank_sum and negative_rank_sum values for STAT3
    actual_rank_sum = min(actual_rank_sum_df[actual_rank_sum_df['Symbols'] == 'STAT3']['actual_rank_sum'].values[0],
                          actual_rank_sum_df[actual_rank_sum_df['Symbols'] == 'STAT3'][
                              'actual_negative_rank_sum'].values[0])
    # Plot a vertical line at actual rank_sum value for STAT3
    plt.axvline(x=actual_rank_sum, color='r', linestyle='--')
    # save the plot
    plt.savefig('../data/stat3_rank_sum.png')
    plt.show()


    # Plot jun_rank_sum_list in histogram
    plt.hist(jun_rank_sum_list, bins=100, edgecolor='white', color='#607c8e')
    plt.title('Histogram of rank-sum for JUN')
    plt.xlabel('Rank-sum')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    # Find actual rank_sum value for JUN
    # actual_rank_sum = actual_rank_sum_df[actual_rank_sum_df['Symbols'] == 'JUN']['actual_rank_sum'].values[0]
    # Find the value which is minimum between rank_sum and negative_rank_sum values for JUN
    actual_rank_sum = min(actual_rank_sum_df[actual_rank_sum_df['Symbols'] == 'JUN']['actual_rank_sum'].values[0],
                          actual_rank_sum_df[actual_rank_sum_df['Symbols'] == 'JUN']['actual_negative_rank_sum'].values[0])
    # Plot a vertical line at actual rank_sum value for JUN
    plt.axvline(x=actual_rank_sum, color='r', linestyle='--')
    # save the plot
    plt.savefig('../data/jun_rank_sum.png')
    plt.show()


    # Plot irf1_rank_sum_list in histogram
    plt.hist(irf1_rank_sum_list, bins=100, edgecolor='white', color='#607c8e')
    plt.title('Histogram of rank-sum for IRF1')
    plt.xlabel('Rank-sum')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    # Find actual rank_sum value for MYC
    # actual_rank_sum = actual_rank_sum_df[actual_rank_sum_df['Symbols'] == 'MYC']['actual_rank_sum'].values[0]
    # Find the value which is minimum between rank_sum and negative_rank_sum values for MYC
    actual_rank_sum = min(actual_rank_sum_df[actual_rank_sum_df['Symbols'] == 'IRF1']['actual_rank_sum'].values[0],
                          actual_rank_sum_df[actual_rank_sum_df['Symbols'] == 'IRF1'][
                              'actual_negative_rank_sum'].values[0])
    # Plot a vertical line at actual rank_sum value for MYC
    plt.axvline(x=actual_rank_sum, color='r', linestyle='--')
    # save the plot
    plt.savefig('../data/irf1_rank_sum.png')
    plt.show()


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
    priors_file = '../data/causal-priors.txt'  # sys.argv[1]
    diff_file = '../data/differential-exp.tsv'  # sys.argv[2]

    main(priors_file, diff_file)
