import argparse
import os

import pandas as pd
import numpy as np
import sys
import requests
from scipy.stats import zscore
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import statsmodels.stats.multitest as smm
from matplotlib.backends.backend_pdf import PdfPages


def run_bh(p_value_file):
    pValDf = pd.read_csv(p_value_file, sep='\t', index_col=0)
    pValDf = np.abs(pValDf)

    adjustedPVals = pd.DataFrame(columns=pValDf.columns, index=pValDf.index)
    actRej = pd.DataFrame(columns=pValDf.columns, index=pValDf.index)

    for (tf, pVals) in pValDf.items():
        pVals = pVals.dropna()
        pV_df = pd.DataFrame(index=pVals.index, columns=['pVal'], data=pVals.values)
        pV_df.sort_values(by='pVal', inplace=True)

        reject, adj_p_val = smm.fdrcorrection(pV_df['pVal'].values, method='i', is_sorted=True, alpha=0.1)
        pV_df[tf] = adj_p_val

        # Insert pV_df into adjustedPVals by column by index
        adjustedPVals.loc[pV_df.index.values, tf] = pV_df[tf].values
        actRej.loc[pV_df.index.values, tf] = reject

    return adjustedPVals, actRej


def main(umap_file, p_value_file, dest_file):
    adjP, acptRej = run_bh(p_value_file)
    umap_df = pd.read_csv(umap_file, sep='\t')

    figures = []

    # Run loop through columns and index of acptRej
    for (tf, cells) in acptRej.items():
        print(tf)
        print(cells)
        plt.figure(figsize=(10, 10))
        plt.scatter()
        plt.title(tf)
        figures.append(plt.gcf())

    with PdfPages(dest_file) as pdf:
        for fig in figures:
            pdf.savefig(fig)

    plt.close('all')


if __name__ == '__main__':
    # Get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-coord', '--coordinate-file', help='UMAP coordinate file path', required=True)
    parser.add_argument('-pval', '--p-value-file', help='P value file path', required=True)
    parser.add_argument('-dest', '--destination-pdf-file', help='Path of destination pdf file', required=True)

    args = parser.parse_args()
    print(args)

    # Check if coordinate file exists
    if not os.path.isfile(args.coordinate_file):
        print("Coordinate file/UMAP coordinate file does not exist.")
        sys.exit(1)

    # Check if p value file exists
    if not os.path.isfile(args.p_value_file):
        print('P value file does not exist.')
        sys.exit(1)

    # Run the main file
    main(args.coordinate_file, args.p_value_file, args.destination_pdf_file)
