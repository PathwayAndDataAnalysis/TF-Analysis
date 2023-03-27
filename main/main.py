import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    cp = pd.read_csv('../data/causal-priors.txt', sep='\t', header=None)
    cp.columns = ['Symbols', 'action', 'targetSymbol', 'reference', 'residual']

    # remove all columns except upregulates-expression and downregulates-expression
    cp = cp[cp['action'].isin(['upregulates-expression', 'downregulates-expression'])]
    # reset index
    cp = cp.reset_index(drop=True)

    # delete reference and residual columns
    cp = cp.drop(['reference', 'residual'], axis=1)

    de = pd.read_csv('../data/differential-exp.tsv', sep='\t')

    # Create a new column named updown based on positive
    # and negative values of SignedP column of de dataframe
    de['updown'] = np.where(de['SignedP'] > 0, '1', '-1')

    # Sort SignedP column in ascending order if updown column is 1
    # and sort absolute values of SignedP column in ascending order if updown column is -1
    de = de.sort_values(by=['updown', 'SignedP'], ascending=[False, True])

    # Add new column named rank to de dataframe
    de['rank'] = np.arange(len(de))

    # Add reverse_rank column to de dataframe
    de['reverse_rank'] = de['rank'].max() - de['rank']


    print(cp.head())
    print(de.head())


if __name__ == '__main__':
    main()
