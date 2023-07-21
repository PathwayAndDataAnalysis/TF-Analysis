import pandas as pd
import numpy as np
from scipy.stats import zscore
from joblib import Parallel, delayed
from timeit import default_timer as timer

cp_file = '../../data/causal-priors.txt'
sde_file = '../../data/mouse_to_human_normalized5k.tsv'

cpo_df = pd.read_csv(cp_file, sep='\t', header=None, usecols=[0, 1, 2], names=['symbol', 'action', 'targetSymbol'])
cpo_df = cpo_df[cpo_df['action'].isin(['upregulates-expression', 'downregulates-expression'])]
cpo_df.reset_index(drop=True, inplace=True)
cpo_df['isUp'] = np.where(cpo_df['action'] == 'upregulates-expression', 1, -1)
cpo_df.drop(['action'], axis=1, inplace=True)

sde_df = pd.read_csv(sde_file, sep='\t', header=0, index_col=0).T
sde_df.replace(0, np.nan, inplace=True)
sde_df = pd.DataFrame(zscore(sde_df, nan_policy='omit'), index=sde_df.index, columns=sde_df.columns)
print("Files read complete...")

# There may be some targetSymbols in cpo_df which are not present in sde_df
# Now remove those rows from cpo_df
cpo_df = cpo_df[cpo_df['targetSymbol'].isin(sde_df.columns)]
cpo_df.reset_index(drop=True, inplace=True)

cpo_grouped_df = cpo_df.groupby('symbol')['isUp'].apply(list).reset_index(name='upDownList')
cpo_grouped_df['targetList'] = cpo_df.groupby('symbol')['targetSymbol'].apply(list).reset_index(name='targetList')[
    'targetList']
cpo_grouped_df['upDownCount'] = cpo_grouped_df['upDownList'].apply(lambda x: len(x))
max_target = np.max(cpo_grouped_df['upDownCount'])

# load npz file to numpy array
distribution = np.load('data/distribution.npz')['distribution']

iters = 1_000_000


def cell_worker(row):
    cell = pd.DataFrame({'symbol': row.index, 'zscore': row.values})
    # Remove rows of cell if zscore is nan
    cell.dropna(inplace=True)
    cell.reset_index(drop=True, inplace=True)
    cell.sort_values(by=['zscore'], ascending=[False], inplace=True)
    cell.reset_index(drop=True, inplace=True)

    cell['acti_rank'] = cell.index
    cell['acti_rank'] = (cell['acti_rank'] + 0.5) / len(cell)
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
    cpo_df_c_grouped['pos_rs'] = cpo_df_c.groupby('symbol')['pos_rs'].apply(list).reset_index(name='pos_rs')['pos_rs']
    cpo_df_c_grouped['neg_rs'] = cpo_df_c.groupby('symbol')['neg_rs'].apply(list).reset_index(name='neg_rs')['neg_rs']
    cpo_df_c_grouped['rs'] = cpo_df_c_grouped.apply(
        lambda x: np.min([np.mean(x['pos_rs']), np.mean(x['neg_rs'])]), axis=1
    )
    cpo_df_c_grouped['rs'] = cpo_df_c_grouped.apply(
        lambda x: x['rs'] if np.mean(x['pos_rs']) > np.mean(x['neg_rs']) else -1 * x['rs'], axis=1
    )

    # Counting how many times the RS is less than the random distribution
    for idx, row in cpo_df_c_grouped.iterrows():
        arr = distribution[row['upDownCount'] - 1]
        rs_count = np.searchsorted(arr, row['rs'])
        if rs_count == 0:
            rs_count = rs_count + 1
        if np.sign(row['rs']) == -1:
            rs_count = -1 * rs_count
        cpo_df_c_grouped.loc[idx, 'count'] = rs_count
        cpo_df_c_grouped.loc[idx, 'p-value'] = rs_count / iters

    return np.zeros(8)


# # Get the top 5 rows of sde_df to create another dataframe
# t10df = sde_df.head(5)
# t10df.reset_index(drop=True, inplace=True)
# print("Type of t10df: ", type(t10df))
#
# for idx, row in t10df.iterrows():
#     print("Type of row: ", type(row))
#     cell_worker(row)

output = Parallel(n_jobs=-1)(delayed(cell_worker)(row) for idx, row in sde_df.head(5).iterrows())
print(output)
