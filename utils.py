import os

import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np


label = ['Stage']
used_features = ['Power [鎂齗', 'BetaRel [%]', 'AlphaRel [%]', 'ThetaRel [%]', 
    'DeltaRel [%]', '95% quantile [Hz]', '50% quantile [Hz]', 'BSRShort C1',
    'BSRMedium C1', 'aEEGmin C1 [鎂]', 'aEEGmax C1 [鎂]']


def load_data(path='./data/narcotrend_all'):
    filenames = os.listdir(path)
    df_narco_all = []
    # Merge usable data
    for filename in filenames:
        df_narco = pd.read_excel(os.path.join(path, filename))
        columns = df_narco.iloc[2]
        df_narco = df_narco.iloc[3:]
        df_narco.columns = columns
        # drop NaN stage and fill NaN feature with 0  
        df_narco = df_narco[df_narco['Stage'].notnull()].fillna(0)
        df_narco_all.append(df_narco)
    df_narco_all = pd.concat(df_narco_all, axis=0)
    label_dict = {k:i for i, k in enumerate(df_narco_all.Stage.unique())}
    print(f'label number {len(label_dict)}')
    df_narco_all['Stage'] = df_narco_all['Stage'].replace(label_dict)
    narco_features, narco_labels  = df_narco_all[used_features], df_narco_all[label]
    print(f'feature number {len(narco_features)}')
    return narco_features, narco_labels


def nacro_train_test_split(path='./data/narcotrend_all', test_size=0.2, random_state=0):
    narco_features, narco_labels = load_data(path)
    X_train, X_test, y_train, y_test = train_test_split(narco_features, narco_labels, test_size=test_size, random_state=random_state)
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train).reshape(-1), np.array(y_test).reshape(-1)
    return X_train, X_test, y_train, y_test