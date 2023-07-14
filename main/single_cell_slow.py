import sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from scipy.stats import zscore


def main(cp_file: str, sde_file: str, iters: int):
    """
    :param cp_file: File path of causal-priors file
    :param sde_file:
    :param iters: Number of iterations
    """
    if cp_file is None or sde_file is None:
        print('Please provide causal-priors and differential-exp file path')
        sys.exit(1)

    if not isinstance(cp_file, str) or not isinstance(sde_file, str):
        print('Please provide causal-priors and differential-exp file path as string')
        sys.exit(1)

    try:
        print("Reading files...")
        cpo_df = pd.read_csv(cp_file, sep='\t', header=None, usecols=[0, 1, 2],
                             names=['Symbols', 'action', 'targetSymbol'], squeeze=True)
        cpo_df = cpo_df[cpo_df['action'].isin(['upregulates-expression', 'downregulates-expression'])]
        cpo_df.reset_index(drop=True, inplace=True)
        cpo_df['isUp'] = np.where(cpo_df['action'] == 'upregulates-expression', 1, -1)
        cpo_df.drop(['action'], axis=1, inplace=True)

        sde_df = pd.read_csv(sde_file, sep='\t', header=0, index_col=0).T
        # sde_df = pd.read_csv(sde_file, sep='\t', header=0, index_col=0)
        sde_df.replace(0, np.nan, inplace=True)
        sde_df = pd.DataFrame(zscore(sde_df, nan_policy='omit'), index=sde_df.index, columns=sde_df.columns)
        sde_df.columns = sde_df.columns.str.upper()
        # Remove columns of sde_df if is not present in targetSymbol column of cpo_df dataframe
        sde_df = sde_df[sde_df.columns.intersection(cpo_df['targetSymbol'])]
        print("Files read complete...")

        pValueDf = pd.DataFrame(columns=np.unique(cpo_df['Symbols']), index=sde_df.index)

        # Each row is one differential expression
        count = 0
        for idx, row in sde_df.iterrows():
            cell_timer = timer()
            cp_df = cpo_df.copy()
            cell_de = pd.DataFrame()

            row = row.dropna()
            cell_de['Symbols'] = row.index
            cell_de['zscore'] = row.values

            # Create a new column named updown based on positive
            # and negative values of zscore column of de_df dataframe
            cell_de['updown'] = np.where(cell_de['zscore'] > 0, '1', '-1')
            # Sort zscore column in ascending order if updown column is 1
            # and sort absolute values of zscore column in ascending order if updown column is -1
            cell_de.sort_values(by=['zscore'], ascending=[False], inplace=True)
            cell_de.reset_index(drop=True, inplace=True)

            # Remove rows of cp_df dataframe if targetSymbol is not present in Symbols column of rank_df dataframe
            cp_df = cp_df[cp_df['targetSymbol'].isin(cell_de['Symbols'])]
            cp_df = cp_df.reset_index(drop=True)

            # Find the max rank
            max_rank = len(cell_de) - 1
            cell_de['rank'] = np.arange(max_rank + 1)
            cell_de['reverse_rank'] = max_rank - cell_de['rank']

            # Find the rank and reverse rank of targetSymbols
            cp_df = cp_df.merge(cell_de[['Symbols', 'rank']], left_on='targetSymbol', right_on='Symbols', how='left')
            cp_df['revRank'] = max_rank - cp_df['rank']
            cp_df.drop('Symbols_y', axis=1, inplace=True)
            cp_df.rename(columns={'Symbols_x': 'Symbols'}, inplace=True)

            # Calculate the positive rank sum
            pos_rank_sum = np.where(cp_df['isUp'] == 1, cp_df['rank'], cp_df['revRank'])
            pos_sum_df = cp_df.groupby('Symbols')[['Symbols', 'isUp']].apply(
                lambda x: np.sum(pos_rank_sum[x.index])).reset_index(name='posRS')
            # Calculate the negative rank sum
            neg_rank_sum = np.where(cp_df['isUp'] == -1, cp_df['rank'], cp_df['revRank'])
            neg_sum_df = cp_df.groupby('Symbols')[['Symbols', 'isUp']].apply(
                lambda x: np.sum(neg_rank_sum[x.index])).reset_index(name='negRS')
            # Merge the positive and negative rank sums
            rank_sum_df = pos_sum_df.merge(neg_sum_df, on='Symbols')

            rank_sum_df['RS'] = np.where(rank_sum_df['posRS'] < rank_sum_df['negRS'], rank_sum_df['posRS'],
                                         rank_sum_df['negRS'])
            rank_sum_df['whichRS'] = np.where(rank_sum_df['posRS'] < rank_sum_df['negRS'], '+1', '-1')
            rank_sum_df.drop(['posRS', 'negRS'], axis=1, inplace=True)
            rank_sum_df['RS'] = rank_sum_df['RS'] * rank_sum_df['whichRS'].astype(int)

            # Create a new dataframe with key: Symbols and value: List of 1's and -1's
            cp_df_grouped = cp_df.groupby('Symbols')['isUp'].apply(list).reset_index(name='upDownList')
            cp_df_grouped['targetList'] = cp_df.groupby('Symbols')['targetSymbol'].apply(
                list).reset_index(name='targetList')['targetList']
            # Count the number of 1's and -1's in upDownList column and store it in a new column
            cp_df_grouped['upDownCount'] = cp_df_grouped['upDownList'].apply(lambda x: len(x))
            cp_df_grouped = cp_df_grouped[cp_df_grouped['upDownCount'] >= 3]

            # Find column RS, whichRS from rank_sum_df and merge it with output_df dataframe
            cp_df_grouped = cp_df_grouped.merge(rank_sum_df[['Symbols', 'RS', 'whichRS']], on='Symbols', how='left')

            # Find maximum number of targets for a Symbol
            max_target = np.max(cp_df_grouped['upDownCount'])

            uniqueUpDowns = cp_df_grouped['upDownCount'].unique()

            # Initialize the array to store results
            result = np.zeros((len(uniqueUpDowns), iters))

            loop_start = timer()
            for i in range(iters):
                drawn_list = np.random.choice(max_rank + 1, max_target, replace=False)
                reverse_drawn_list = max_rank - drawn_list
                min_rank_sum = np.minimum(np.cumsum(drawn_list), np.cumsum(reverse_drawn_list))
                result[:, i] = [min_rank_sum[x - 1] for x in uniqueUpDowns]
            loop_end = timer()
            print(f'Random Distribution Duration: {loop_end - loop_start}')

            # Merge all columns of results_array into a list of values
            result = pd.DataFrame(result).apply(lambda x: x.values.tolist(), axis=1)
            df_result = pd.concat([pd.DataFrame(uniqueUpDowns), result], axis=1)
            df_result.columns = ['totalCount', 'rank_sum_list']

            # Merge output_df and df_result dataframe
            cp_df_grouped['RSList'] = cp_df_grouped['upDownCount'].apply(
                lambda x: df_result[df_result['totalCount'] == x]['rank_sum_list'].values[0])
            cp_df_grouped['pValue'] = cp_df_grouped.apply(
                lambda x: np.sum(x['RSList'] <= np.abs(x['RS'])) / (iters * np.sign(x['RS'])), axis=1)
            # Handle zero pValues
            cp_df_grouped['pValue'] = np.where(cp_df_grouped['pValue'] == 0, 1 / iters, cp_df_grouped['pValue'])

            pValueDf.loc[idx, cp_df_grouped['Symbols']] = cp_df_grouped['pValue'].values
            print('Transcription factor count: ', len(cp_df_grouped['Symbols'].unique()))
            print(f'Cell id {idx}, Cell count {count} completed and took {timer() - cell_timer} seconds \n')
            count += 1

        # Remove columns which has all NaN values
        pValueDf = pValueDf.dropna(axis=1, how='all')
        pValueDf.fillna(0.0, inplace=True)
        pValueDf.to_csv('../output/pValSlw5k.tsv', sep='\t', index=True)

    except Exception as e:
        print('Exception: ', type(e), e.args, e)
        print(e)
        sys.exit(1)


if __name__ == '__main__':
    priors_file = '../data/causal-priors.txt'
    single_cell_file = '../data/5knormalized_mat.tsv'

    start = timer()
    main(priors_file, single_cell_file, iters=10_000)
    end = timer()
    print("Time taken: ", end - start)
