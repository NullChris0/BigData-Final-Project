from saver import dATA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def drop_missings(num):
    # make a copy for our target trainset, for combined data
    if 'SalePrice' in dATA.data.columns:
        sale_price = dATA.data['SalePrice']
    else:
        sale_price = None

    null_sum = dATA.data.isnull().sum()
    dATA.data.drop(columns=dATA.data.columns[null_sum > len(dATA.data) * num], inplace=True)
    # give it back!
    if sale_price is not None:
        dATA.data['SalePrice'] = sale_price

    return dATA.info, dATA.find_string


def numeralization(string, apply_log=False, drop_columns=False):
    """
    Convert non-numeric columns to numeric format where possible, optionally applying log transformation,
    and optionally dropping columns that cannot be converted or as specified.

    Parameters:
        string: The target column names to be convert.
        apply_log (bool): Whether to apply log10 transformation to numeric converted columns.
        drop_columns (bool): Whether to drop the columns that cannot be converted to numeric.

    Returns:
        Dataframe's info and find_string.
    """
    select = [c.strip() for c in string.split(",") if c.strip()]  # Remove whitespace and split by comma
    selected = dATA.data.columns.intersection(select)
    columns_to_drop = []  # Track columns that were dropped

    for column in selected:
        # Attempt to convert to numeric type
        converted = pd.to_numeric(dATA.data.loc[:, column].replace(r'[^\d.,]+', '', regex=True).replace(r'[$,-]', '', regex=True),errors='ignore')

        if converted.dtype == 'object':
            # If all values are NaN, conversion failed; consider dropping if specified
            if drop_columns:
                columns_to_drop.append(column)
        else:
            # Apply log transformation if specified
            if apply_log:
                dATA.data[column] = np.log10(converted + 1)
                dATA.data[column] = dATA.data[column].astype(float)
            elif drop_columns:
                columns_to_drop.append(column)
            else:
                dATA.data[column] = converted

    # Drop columns that could not be converted if drop_columns is True
    if drop_columns:
        dATA.data.drop(columns=columns_to_drop, inplace=True)

    return dATA.info, dATA.find_string
    
def change_range(min, max, targets):
    for i in targets:
        dATA.data = dATA.data[(dATA.data[i] >= min ) & (dATA.data[i] <= max )]
        dATA.data[i].astype(float)
    return dATA.info

def clear2full():
    # 去除重复行
    dATA.data.drop_duplicates(inplace=True)

    # 初筛，这些是包含缺失值的列名
    missings = dATA.data.columns[dATA.data.isnull().any()]
    # 利用上面的索引过滤出缺失&数值列
    miss_num = dATA.data[missings].select_dtypes('number').columns
    # 执行填充
    for i in miss_num:
        dATA.data[i] = dATA.data[i].fillna(dATA.data[i].mean())

    # 非数值型数据
    miss_str = dATA.data[missings].select_dtypes('object').columns
    for i in miss_str:
        freq = dATA.data[i].mode().iloc[0]
        dATA.data[i] = dATA.data[i].fillna(freq)

    return dATA.info, dATA.find_string

def cal_normal(features):
    X_train = dATA.data[features].values
    # 标准化器
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_train)
    for i, feature in enumerate(features):
        dATA.data[feature] = X_norm[:, i]
    
    return X_train, X_norm