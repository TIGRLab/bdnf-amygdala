#!/usr/bin/env python

import os, sys
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind


# compare methods for a given link
TESTS = {'amyg': ['R','R'],
        'ctx':   ['Q23_RAmy_LCT_maskq10_STG.txt', 'Q23_RAmy_LCT_maskq10_Ins.txt']}

# load data
data = pd.read_csv('results_transposed_cc.csv')
demo = pd.read_csv('demographics.csv')
merged = pd.merge(data, demo[['ID', 'MDD']], left_on='subj', right_on='ID')

# visualize similarities / differences between methods
subset_pca = merged[(merged.method == 'pca')]
subset_cc = merged[(merged.method == 'cc')]
subset_corr = merged[(merged.method == 'corr')]

plt.imshow(np.corrcoef(np.vstack((subset_corr.value,
                                  subset_pca.value,
                                  subset_cc.value))), interpolation='nearest')


# run 2 sample t-tests for each
tests = [TESTS[k] for k in TESTS.keys()]

for t in range(len(tests[0])):

    subset = merged[(merged.amyg == tests[0][t]) & (merged.ctx_roi == tests[1][t])]
    idx_cc = subset.method == 'cc'
    idx_pc = subset.method == 'pca'
    idx_r  = subset.method == 'corr'

    subset_cc = subset.loc[idx_cc]
    subset_pc = subset.loc[idx_pc]
    subset_r = subset.loc[idx_r]

    ax = sns.boxplot(x='method', y='value', hue='MDD', data=subset, linewidth=2.5)
    sns.plt.show()

    cc_results = ttest_ind(np.asarray(subset_cc.loc[subset_cc.MDD == 'no', 'value']),
              np.asarray(subset_cc.loc[subset_cc.MDD == 'yes', 'value']))
    pca_results = ttest_ind(np.asarray(subset_pc.loc[subset_pc.MDD == 'no', 'value']),
              np.asarray(subset_pc.loc[subset_pc.MDD == 'yes', 'value']))
    corr_results = ttest_ind(np.asarray(subset_r.loc[subset_r.MDD == 'no', 'value']),
              np.asarray(subset_r.loc[subset_r.MDD == 'yes', 'value']))

    print(cc_results)
    print(pca_results)
    print(corr_results)

