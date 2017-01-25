#!/usr/bin/env python

import os, sys
import pandas as pd
import seaborn as sns

data = pd.read_csv('results.csv')
demo = pd.read_csv('demographics.csv')

merged = pd.merge(data, demo[['ID', 'MDD']], left_on='subj', right_on='ID')

values = {'amyg': ['L', 'R'],
       'ctx_roi': ['Q23_RAmy_LCT_maskq10_Ins.txt']}

subset = merged[(merged.amyg == 'L') & (merged.ctx_roi == 'Q23_RAmy_LCT_maskq10_CUN.txt')]
idx_cc = subset.method == 'cc'
idx_pc = subset.method == 'pca'
idx_r  = subset.method == 'corr'

subset.loc[idx_cc, 'value']  = subset.loc[idx_cc, 'value'] - np.mean(subset.loc[idx_cc, 'value'])
subset.loc[idx_pc, 'value']  = subset.loc[idx_pc, 'value'] - np.mean(subset.loc[idx_pc, 'value'])
subset.loc[idx_r, 'value']  = subset.loc[idx_r, 'value'] - np.mean(subset.loc[idx_r, 'value'])

ax = sns.boxplot(x='method', y='value', hue='MDD', data=subset, linewidth=2.5)


