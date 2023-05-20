import sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import statsmodels.stats.multitest as smm


def main(cp_file: str, sde_file: str, iters: int):
    """
    :param cp_file: File path of causal-priors file
    :param de_file: File path of single cell differential expression file
    :param iters: Number of iterations
    """
    if cp_file is None or sde_file is None:
        print('Please provide causal-priors and differential-exp file path')
        sys.exit(1)

    if not isinstance(cp_file, str) or not isinstance(sde_file, str):
        print('Please provide causal-priors and differential-exp file path as string')
        sys.exit(1)

    try:
        cpo_df = pd.read_csv(cp_file, sep='\t', header=None)
        cpo_df.columns = ['Symbols', 'action', 'targetSymbol', 'reference', 'residual']
        cpo_df = cpo_df.drop(['reference', 'residual'], axis=1)
        cpo_df = cpo_df[cpo_df['action'].isin(['upregulates-expression', 'downregulates-expression'])]
        cpo_df = cpo_df.reset_index(drop=True)

        sde_df = pd.read_csv(sde_file, sep='\t', index_col=0)

        # Find the columns which has comma separated values
        comma_cols = sde_df.columns[sde_df.columns.str.contains(",")]

        new_cols_df = pd.DataFrame()
        for col in comma_cols:
            col_sp = col.split(",")
            new_cols = pd.concat([sde_df[col]] * len(col_sp), axis=1)
            new_cols.columns = col_sp
            new_cols_df = pd.concat([new_cols_df, new_cols], axis=1)

        # Drop the columns which has comma in the name
        sde_df.drop(comma_cols, axis=1, inplace=True)
        sde_df = pd.concat([sde_df, new_cols_df], axis=1)

        # Remove all the columns whose mean is 0 except the first column
        sde_df = sde_df.loc[:, (sde_df != 0).any(axis=0)]
        # Remove all the rows whose mean is 0
        sde_df = sde_df.loc[(sde_df != 0).any(axis=1)]

        # Normalize the data using z-score
        sde_df = sde_df.apply(lambda x: (x - np.mean(x)) / np.std(x), axis=0)

        count = 0

        # Each row is one differential expression
        for idx, row in sde_df.iterrows():
            cp_df = cpo_df.copy()
            cell_de = pd.DataFrame()
            cell_de['Symbols'] = sde_df.columns
            cell_de['SignedP'] = row.values

            # ----------------------------------------------------------------
            # Create a new column named updown based on positive
            # and negative values of SignedP column of de_df dataframe
            cell_de['updown'] = np.where(cell_de['SignedP'] > 0, '1', '-1')

            # Sort SignedP column in ascending order if updown column is 1
            # and sort absolute values of SignedP column in ascending order if updown column is -1
            cell_de = cell_de.sort_values(by=['updown', 'SignedP'], ascending=[False, True])

            # Remove rows of cp_df dataframe if targetSymbol is not present in Symbols column of rank_df dataframe
            cp_df = cp_df[cp_df['targetSymbol'].isin(cell_de['Symbols'])]
            # Reset index
            cp_df = cp_df.reset_index(drop=True)

            # Find the max rank
            max_rank = len(cell_de) - 1
            cell_de['rank'] = np.arange(max_rank + 1)
            cell_de['reverse_rank'] = max_rank - cell_de['rank']

            # Represent upregulates-expression as 1 and downregulates-expression as -1
            cp_df['isUp'] = np.where(cp_df['action'] == 'upregulates-expression', 1, -1)
            cp_df.drop(['action'], axis=1, inplace=True)

            # Find the rank and reverse rank of targetSymbols
            cp_df['rank'] = cp_df['targetSymbol'].apply(lambda x: cell_de[cell_de['Symbols'] == x]['rank'].values[0])
            cp_df['revRank'] = max_rank - cp_df['rank']

            # Calculate the actual rank sum of each Symbol for positive and negative ranks
            rank_sum_df = cp_df.groupby('Symbols').apply(
                lambda x: np.concatenate((x[x['isUp'] == 1]['rank'], x[x['isUp'] == -1]['revRank']))
            ).reset_index(name='posRSList')
            rank_sum_df['negRSList'] = cp_df.groupby('Symbols').apply(
                lambda x: np.concatenate((x[x['isUp'] == 1]['revRank'], x[x['isUp'] == -1]['rank']))
            ).reset_index(name='negRSList')['negRSList']

            # Chose value between column posRS and negRS and also find column chosen and store it in a new column
            rank_sum_df['posRS'] = rank_sum_df['posRSList'].apply(lambda x: np.sum(x))
            rank_sum_df['negRS'] = rank_sum_df['negRSList'].apply(lambda x: np.sum(x))
            rank_sum_df['RS'] = np.where(rank_sum_df['posRS'] < rank_sum_df['negRS'], rank_sum_df['posRS'],
                                         rank_sum_df['negRS'])
            rank_sum_df['whichRS'] = np.where(rank_sum_df['posRS'] < rank_sum_df['negRS'], '+1', '-1')
            rank_sum_df['whichRSList'] = np.where(rank_sum_df['posRS'] < rank_sum_df['negRS'], rank_sum_df['posRSList'],
                                                  rank_sum_df['negRSList'])

            # Create a new dataframe with key: Symbols and value: List of 1's and -1's
            output_df = cp_df.groupby('Symbols')['isUp'].apply(list).reset_index(name='upDownList')
            output_df['targetList'] = \
            cp_df.groupby('Symbols')['targetSymbol'].apply(list).reset_index(name='targetList')[
                'targetList']
            # Count the number of 1's and -1's in upDownList column and store it in a new column
            output_df['upDownCount'] = output_df['upDownList'].apply(lambda x: len(x))
            # Remove rows from output_df dataframe if upDownCount is less than 3
            output_df = output_df[output_df['upDownCount'] >= 3]

            # Find column RS, whichRS from rank_sum_df and merge it with output_df dataframe
            output_df = output_df.merge(rank_sum_df[['Symbols', 'RS', 'whichRS', 'whichRSList']], on='Symbols',
                                        how='left')

            # Find maximum number of targets for a Symbol
            max_target = np.max(output_df['upDownCount'])

            output_df['rankLessThanActual'] = 0
            uniqueUpDowns = output_df['upDownCount'].unique()

            # Initialize the array to store results
            result = np.zeros((len(uniqueUpDowns), iters))

            loop_start = timer()
            for i in range(iters):
                # Draw a random number from 0 to max_rank+1
                drawn_list = np.random.choice(max_rank + 1, max_target, replace=False)
                reverse_drawn_list = max_rank - drawn_list
                min_rank_sum = np.minimum(np.cumsum(drawn_list), np.cumsum(reverse_drawn_list))
                result[:, i] = [min_rank_sum[x - 1] for x in uniqueUpDowns]

            loop_end = timer()
            print(f'Loop took {loop_end - loop_start} seconds')

            # Merge all columns of results_array into a list of values
            result = pd.DataFrame(result).apply(lambda x: x.values.tolist(), axis=1)
            df_result = pd.concat([pd.DataFrame(uniqueUpDowns), result], axis=1)
            df_result.columns = ['totalCount', 'rank_sum_list']

            # Merge output_df and df_result dataframe
            output_df['RSList'] = output_df['upDownCount'].apply(
                lambda x: df_result[df_result['totalCount'] == x]['rank_sum_list'].values[0])

            # Count the number of times RS is less than RSList
            output_df['rankLessThanActual'] = output_df.apply(lambda x: np.sum(np.array(x['RSList']) < x['RS']), axis=1)
            output_df['rankLessThanActual'] = 1 + output_df['rankLessThanActual']
            output_df.drop(['RSList'], axis=1, inplace=True)

            # Calculate the p-value
            output_df['pValue'] = output_df['rankLessThanActual'] / iters

            # Sort rows by pValue
            output_df.sort_values(by=['pValue'], inplace=True)

            # Calculate the FDR Benjamini-Hochberg
            reject, pvals_corrected, _, _ = smm.multipletests(output_df['pValue'], alpha=0.05, method='fdr_bh')
            output_df['pValueCorrected'] = pvals_corrected

            # Save the output dataframe to a file
            output_df.to_csv('../output/single_analysis_output_' + str(idx) + '.tsv', sep='\t', index=False)

            count += 1
            print(f'Iteration {count} completed')
            if count == 3:
                break

    except FileNotFoundError:
        print('File not found: ')
        sys.exit(1)

    except Exception as e:
        print('Exception: ', type(e), e.args, e)
        print(e)
        sys.exit(1)


if __name__ == '__main__':
    priors_file = '../data/causal-priors.txt'
    single_cell_file = '../data/tf_scores_t.tsv'

    start = timer()
    main(priors_file, single_cell_file, iters=10_000)
    end = timer()
    print("Time taken: ", end - start)
