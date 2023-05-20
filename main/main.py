import sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import statsmodels.stats.multitest as smm


def main(cp_file: str, de_file: str, iters: int):
    """
    :param cp_file: File path of causal-priors file
    :param de_file: File path of differential-exp file
    :param iters: Number of iterations
    """
    if cp_file is None or de_file is None:
        print('Please provide causal-priors and differential-exp file path')
        sys.exit(1)

    if not isinstance(cp_file, str) or not isinstance(de_file, str):
        print('Please provide causal-priors and differential-exp file path as string')
        sys.exit(1)

    try:
        cp_df = pd.read_csv(cp_file, sep='\t', header=None)
        cp_df.columns = ['Symbols', 'action', 'targetSymbol', 'reference', 'residual']
        cp_df = cp_df.drop(['reference', 'residual'], axis=1)
        cp_df = cp_df[cp_df['action'].isin(['upregulates-expression', 'downregulates-expression'])]
        cp_df = cp_df.reset_index(drop=True)

        de_df = pd.read_csv(de_file, sep='\t')

        # Create a new column named updown based on positive
        # and negative values of SignedP column of de_df dataframe
        de_df['updown'] = np.where(de_df['SignedP'] > 0, '1', '-1')

        # Sort SignedP column in ascending order if updown column is 1
        # and sort absolute values of SignedP column in ascending order if updown column is -1
        de_df = de_df.sort_values(by=['updown', 'SignedP'], ascending=[False, True])

        # Remove rows of cp_df dataframe if targetSymbol is not present in Symbols column of rank_df dataframe
        cp_df = cp_df[cp_df['targetSymbol'].isin(de_df['Symbols'])]
        # Reset index
        cp_df = cp_df.reset_index(drop=True)

        # Find the max rank
        max_rank = len(de_df) - 1
        de_df['rank'] = np.arange(max_rank + 1)
        de_df['reverse_rank'] = max_rank - de_df['rank']

        # Represent upregulates-expression as 1 and downregulates-expression as -1
        cp_df['isUp'] = np.where(cp_df['action'] == 'upregulates-expression', 1, -1)
        cp_df.drop(['action'], axis=1, inplace=True)

        # Find the rank and reverse rank of targetSymbols
        cp_df['rank'] = cp_df['targetSymbol'].apply(lambda x: de_df[de_df['Symbols'] == x]['rank'].values[0])
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
        output_df['targetList'] = cp_df.groupby('Symbols')['targetSymbol'].apply(list).reset_index(name='targetList')[
            'targetList']
        # Count the number of 1's and -1's in upDownList column and store it in a new column
        output_df['upDownCount'] = output_df['upDownList'].apply(lambda x: len(x))
        # Remove rows from output_df dataframe if upDownCount is less than 3
        output_df = output_df[output_df['upDownCount'] >= 3]

        # Find column RS, whichRS from rank_sum_df and merge it with output_df dataframe
        output_df = output_df.merge(rank_sum_df[['Symbols', 'RS', 'whichRS', 'whichRSList']], on='Symbols', how='left')

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
        reject, pvals_corrected, alphacSidak, alphacBonf = smm.multipletests(output_df['pValue'], alpha=0.05, method='fdr_bh')
        output_df['pValueCorrected'] = pvals_corrected

        # Save the output dataframe to a file
        output_df.to_csv('../output/analysis_output.tsv', sep='\t', index=False)

    except FileNotFoundError:
        print('File not found: ', cp_file)
        sys.exit(1)

    except Exception as e:
        print(e)
        sys.exit(1)


# output_df['targetListSorted'] = output_df.apply(
#             lambda x: [de_df[de_df['Symbols'] == y]['rank'].values[0] for y in x['targetList']], axis=1)

if __name__ == '__main__':
    priors_file = '../data/causal-priors.txt'
    diff_file = '../data/differential-exp.tsv'
    # diff_file = '../data/rslp_vs_lum.tsv'

    start = timer()
    main(priors_file, diff_file, iters=50_000)
    end = timer()
    print("Time taken: ", end - start)

# Command line argument
# if __name__ == '__main__':
#     prior_file = sys.argv[1]
#     diff_file = sys.argv[2]
#     iters = int(sys.argv[3])
#
#     start = timer()
#     main(prior_file, diff_file, iters)
#     end = timer()
#     print("Time taken: ", end - start)
