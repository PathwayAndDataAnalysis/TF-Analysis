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
        cp_df = pd.read_csv(cp_file, sep='\t', header=None, usecols=[0, 1, 2],
                            names=['Symbols', 'action', 'targetSymbol'], squeeze=True)
        cp_df = cp_df[cp_df['action'].isin(['upregulates-expression', 'downregulates-expression'])]
        cp_df.reset_index(drop=True, inplace=True)

        sde_df = pd.read_csv(sde_file, sep='\t', header=0, index_col=0).T
        # sde_df = pd.read_csv(sde_file, sep='\t', header=0, index_col=0)

        # Normalize the data using z-score
        sde_df.replace(0, np.nan, inplace=True)
        # Drop rows and column for some threshold
        sde_df.dropna(axis=0, thresh=(len(sde_df.columns) / 14), inplace=True)
        sde_df.dropna(axis=1, thresh=(len(sde_df.index) / 26), inplace=True)
        sde_df.fillna(0, inplace=True)
        print('sde_df shape: ', sde_df.shape)
        sde_df.columns = sde_df.columns.str.upper()  # Convert column names to upper case
        # Calculate z-score
        sde_df = pd.DataFrame(zscore(sde_df, nan_policy='omit'), index=sde_df.index, columns=sde_df.columns)

        # Remove rows of cp_df dataframe if targetSymbol is not present in columns of sde_df dataframe
        cp_df = cp_df[cp_df['targetSymbol'].isin(sde_df.columns)]
        cp_df.reset_index(drop=True, inplace=True)
        # Remove columns of sde_df dataframe if Symbols is not present in targetSymbol column of cp_df dataframe
        sde_df = sde_df[sde_df.columns.intersection(cp_df['targetSymbol'])]
        # Represent upregulates-expression as 1 and downregulates-expression as -1
        cp_df['isUp'] = np.where(cp_df['action'] == 'upregulates-expression', 1, -1)
        cp_df.drop(['action'], axis=1, inplace=True)

        cp_df_grouped = cp_df.groupby('Symbols')['isUp'].apply(list).reset_index(name='updown')
        cp_df_grouped['targetList'] = \
            cp_df.groupby('Symbols')['targetSymbol'].apply(list).reset_index(name='targetList')['targetList']
        cp_df_grouped['count'] = cp_df_grouped['updown'].apply(lambda x: len(x))
        cp_df_grouped = cp_df_grouped[cp_df_grouped['count'] >= 3]

        max_rank = len(sde_df.columns) - 1
        max_target = np.max(cp_df_grouped['count'])

        # Create the distribution of random ranks
        uniqueUpdown = cp_df_grouped['count'].unique()
        result = np.zeros((len(uniqueUpdown), iters))
        distrb_timer = timer()
        for i in range(iters):
            drawn = np.random.choice(max_rank + 1, max_target, replace=False)
            rev_drawn = max_rank - drawn
            min_rs = np.minimum(np.cumsum(drawn), np.cumsum(rev_drawn))
            result[:, i] = [min_rs[x - 1] for x in uniqueUpdown]
        print('Distribution time: ', timer() - distrb_timer)

        pVals = []
        for idx, row in sde_df.iterrows():
            cell_timer = timer()
            cell = pd.DataFrame()
            cell['Symbols'] = row.index
            cell['SignedP'] = row.values

            cell['updown'] = np.where(cell['SignedP'] > 0, '1', '-1')
            cell.sort_values(by=['updown', 'SignedP'], ascending=[False, True], inplace=True)

            cell['rank'] = np.arange(max_rank + 1)
            cell['revRank'] = max_rank - cell['rank']

            # Find the rank and reverse rank of targetSymbols
            rs_cp_df = cp_df.copy()
            rs_cp_df = rs_cp_df.merge(cell[['Symbols', 'rank']], left_on='targetSymbol', right_on='Symbols', how='left')
            rs_cp_df['revRank'] = max_rank - rs_cp_df['rank']
            rs_cp_df.drop('Symbols_y', axis=1, inplace=True)
            rs_cp_df.rename(columns={'Symbols_x': 'Symbols'}, inplace=True)

            # Calculate actual rank sum
            # rsdf = rs_cp_df.groupby('Symbols').apply(
            #     lambda x: np.sum(
            #         np.concatenate((x[x['isUp'] == 1]['rank'], x[x['isUp'] == -1]['revRank'])))).reset_index(
            #     name='posRSSum')
            #
            # rsdf['negRSSum'] = rs_cp_df.groupby('Symbols').apply(
            #     lambda x: np.sum(
            #         np.concatenate((x[x['isUp'] == 1]['revRank'], x[x['isUp'] == -1]['rank'])))).reset_index(
            #     name='negRSSum')['negRSSum']

            # Calculate the positive rank sum
            pos_rank_sum = np.where(rs_cp_df['isUp'] == 1, rs_cp_df['rank'], rs_cp_df['revRank'])
            pos_sum_df = rs_cp_df.groupby('Symbols')[['Symbols', 'isUp']].apply(
                lambda x: np.sum(pos_rank_sum[x.index])).reset_index(name='posRSSum')
            # Calculate the negative rank sum
            neg_rank_sum = np.where(rs_cp_df['isUp'] == -1, rs_cp_df['rank'], rs_cp_df['revRank'])
            neg_sum_df = rs_cp_df.groupby('Symbols')[['Symbols', 'isUp']].apply(
                lambda x: np.sum(neg_rank_sum[x.index])).reset_index(name='negRSSum')
            # Merge the positive and negative rank sums
            rsdf = pos_sum_df.merge(neg_sum_df, on='Symbols')

            # Choose value between column posRS and negRS and also find column chosen and store it in a new column
            rsdf['RS'] = np.where(rsdf['posRSSum'] < rsdf['negRSSum'], rsdf['posRSSum'], rsdf['negRSSum'])
            rsdf['whichRS'] = np.where(rsdf['posRSSum'] < rsdf['negRSSum'], '+1', '-1')
            rsdf.drop(['posRSSum', 'negRSSum'], axis=1, inplace=True)
            rsdf['RS'] = rsdf['RS'] * rsdf['whichRS'].astype(int)

            # Find The RS of Symbols from rsdf dataframe and store it RS column of cp_df_grouped dataframe
            cp_df_grouped_cell = cp_df_grouped.copy()
            cp_df_grouped_cell['RS'] = cp_df_grouped_cell['Symbols'].map(rsdf.set_index('Symbols')['RS'])

            cp_df_grouped_cell['rankSum'] = cp_df_grouped_cell['count'].apply(
                lambda x: result[np.where(uniqueUpdown == x)[0][0], :]
            )
            cp_df_grouped_cell['pValue'] = cp_df_grouped_cell.apply(
                lambda x: np.sum(x['rankSum'] <= np.abs(x['RS'])) / (iters * np.sign(x['RS'])), axis=1
            )

            pVals.append(cp_df_grouped_cell['pValue'].values.tolist())
            print(f'Cell {idx} time: ', timer() - cell_timer)

        pValueDf = pd.DataFrame(pVals, index=sde_df.index, columns=cp_df_grouped['Symbols'].values)

        pValueDf.to_csv('../output/pValueArrayFst.tsv', sep='\t', index=True, header=True)

    except Exception as e:
        print('Exception: ', type(e), e.args, e)
        print(e)
        sys.exit(1)


if __name__ == '__main__':
    priors_file = '../data/causal-priors.txt'
    single_cell_file = '../data/normalized_mat.tsv'

    start = timer()
    main(priors_file, single_cell_file, iters=500_000)
    end = timer()
    print("Total Time taken: ", end - start)
