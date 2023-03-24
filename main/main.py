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

    # Match targetSymbol column of cp to Symbols column of de
    temp = cp[cp['targetSymbol'].isin(de['Symbols'])]
    # reset index
    temp = temp.reset_index(drop=True)



    print(cp.head())
    print(de.head())


if __name__ == '__main__':
    main()
