import pandas as pd
import numpy as np
from scipy.stats import zscore

cp_file = '../../data/causal-priors.txt'
sc_file = '../../data/mouse_to_human_normalized5k.tsv'

# Read prior file
cpo_df = pd.read_csv(cp_file, sep='\t', header=None, usecols=[0, 1, 2], names=['symbol', 'action', 'targetSymbol'])
cpo_df = cpo_df[cpo_df['action'].isin(['upregulates-expression', 'downregulates-expression'])]
cpo_df.reset_index(drop=True, inplace=True)
cpo_df['isUp'] = np.where(cpo_df['action'] == 'upregulates-expression', 1, -1)
cpo_df.drop(['action'], axis=1, inplace=True)
print("Prior file shape", cpo_df.shape)

# Read Single Cell file
sde_df = pd.read_csv(sc_file, sep='\t', header=0, index_col=0).T
sde_df.replace(0, np.nan, inplace=True)
sde_df = pd.DataFrame(zscore(sde_df, nan_policy='omit'), index=sde_df.index, columns=sde_df.columns)
print("Single Cell file shape", sde_df.shape)
print("Files read complete...")

# Remove rows of cpo_df if targetSymbol column is not present in sde_df columns
cpo_df = cpo_df[cpo_df['targetSymbol'].isin(sde_df.columns)]
cpo_df.reset_index(drop=True, inplace=True)
print("Prior file shape after removing rows which are not in single cell file", cpo_df.shape)

# Group the prior file by symbol column
cpo_grouped_df = cpo_df.groupby('symbol')['isUp'].apply(list).reset_index(name='upDownList')
cpo_grouped_df['targetList'] = cpo_df.groupby('symbol')['targetSymbol'].apply(list).reset_index(name='targetList')[
    'targetList']
cpo_grouped_df['upDownCount'] = cpo_grouped_df['upDownList'].apply(lambda x: len(x))
max_target = np.max(cpo_grouped_df['upDownCount'])

# Generate Distribution
n = 10_000
ranks = [(i + 0.5) / n for i in range(1, n)]
distribution = []
iters = 1_000

for target in range(1, max_target + 1):
    arr = []
    for i in range(iters):
        amr = (np.mean(np.random.choice(n, target, replace=False)) - 0.5) / n
        imr = 1 - amr
        arr.append(np.min([amr, imr]))
    distribution.append(arr)

distribution = np.array(distribution)

# For each cell in single cell
cpo_df_c_grouped = None
result = pd.DataFrame(columns=cpo_grouped_df['symbol'].values, index=sde_df.index)
cell_count = 0
for idx, row in sde_df.iterrows():
    cell = pd.DataFrame(row.index, columns=['symbol'])
    cell['zscore'] = row.values
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
    cpo_df_c_grouped['targetList'] = cpo_df_c.groupby('symbol')['targetSymbol'].apply(list).reset_index(
        name='targetList')['targetList']
    cpo_df_c_grouped['upDownCount'] = cpo_df_c_grouped['upDownList'].apply(lambda x: len(x))
    cpo_df_c_grouped['pos_rs'] = cpo_df_c.groupby('symbol')['pos_rs'].apply(list).reset_index(name='pos_rs')[
        'pos_rs']
    cpo_df_c_grouped['neg_rs'] = cpo_df_c.groupby('symbol')['neg_rs'].apply(list).reset_index(name='neg_rs')[
        'neg_rs']
    cpo_df_c_grouped['rs'] = cpo_df_c_grouped.apply(
        lambda x: np.min([np.mean(x['pos_rs']), np.mean(x['neg_rs'])]),
        axis=1
    )
    # if rs is pos_rs then it is up, if rs is neg_rs then it is down. So if rs is neg_rs then we need to multiply it
    # by -1
    cpo_df_c_grouped['rs'] = cpo_df_c_grouped.apply(
        lambda x: x['rs'] if np.mean(x['pos_rs']) > np.mean(x['neg_rs']) else -1 * x['rs'],
        axis=1
    )

    # Counting how many times the RS is less than the random distribution
    for idx1, row1 in cpo_df_c_grouped.iterrows():
        target = row1['upDownCount']
        rs = row1['rs']
        arr = distribution[target - 1]
        count = 0
        for i in arr:
            if np.abs(rs) >= i:
                count += 1

        if count == 0:
            count = count + 1
        if np.sign(rs) == -1:
            count = -1 * count
        cpo_df_c_grouped.loc[idx1, 'count'] = count
        cpo_df_c_grouped.loc[idx1, 'p-value'] = count / iters

    # Collect p-values of all cells
    result.loc[idx, cpo_df_c_grouped['symbol']] = cpo_df_c_grouped['p-value'].values
    print("Cell count: ", cell_count, " out of ", len(sde_df.index))
    cell_count += 1

result.to_csv('result.csv')
