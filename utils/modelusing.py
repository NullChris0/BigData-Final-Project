from autogluon.tabular import TabularPredictor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from saver import dATA

def split_data(*args):
    if len(args) == 1:
        rows = args[0]; date, col = None, None
    elif len(args) == 2:
        date, col = args; rows = None
    if rows:
        dATA.train, dATA.test = (dATA.data[:rows].copy(), dATA.data[rows:].copy())
        return dATA.train.shape, dATA.test.shape
    elif date and col:
        t = datetime.strptime(date, '%Y-%m-%d').date()
        y = t.year; m = t.month; d = t.day 
        test_start, test_end = pd.Timestamp(2021, 2, 15), pd.Timestamp(2021, 3, 1)
        train_start = pd.Timestamp(y, m, d)
        dATA.data[col] = pd.to_datetime(dATA.data[col], errors='coerce')
        # 处理日期，从 `01/22/2013` 的格式转换为 `2013-01-22` 的格式
        # 根据日期划分为训练集合测试集
        dATA.train = dATA.data[(dATA.data[col] >= train_start) & (dATA.data[col] < test_start)]
        dATA.test = dATA.data[(dATA.data[col] >= test_start) & (dATA.data[col] < test_end)]
        return dATA.train.shape, dATA.test.shape

def train_model(target):
    dATA.predictor = TabularPredictor(label=target).fit(dATA.train)
    preds = dATA.predictor.predict(dATA.test.drop(columns=target))
    print(dATA.predictor.feature_importance(dATA.test))
    return (
        rmse(preds, dATA.test[target]), 
        dict_to_long_string(dATA.predictor.evaluate(dATA.test)),
        show_results(dATA.test[target], preds),
        dATA.predictor.leaderboard(dATA.test),
        dATA.predictor.feature_importance(dATA.test)
    )

def load_model(target, path):
    try:
        dATA.predictor = TabularPredictor.load(path)
        preds = dATA.predictor.predict(dATA.test.drop(columns=[target]))
        print(dATA.predictor.feature_importance(dATA.test))
        return (
            rmse(preds, dATA.test[target]),
            dict_to_long_string(dATA.predictor.evaluate(dATA.test)), 
            show_results(dATA.test[target], preds),
            dATA.predictor.leaderboard(dATA.test),
            dATA.predictor.feature_importance(dATA.test)
        )
    except Exception as e:
        print(e)

def train_new(target):
    dATA.predictor = TabularPredictor(label=target).fit(dATA.train)
    t = dATA.test.drop(columns=[target])
    preds = dATA.predictor.predict(t)
    print(dATA.predictor.feature_importance(dATA.test))
    return (
        np.nan, 
        'No Infomation For New Test Data', 
        show_results(None, preds),
        dATA.predictor.leaderboard(dATA.test),
        dATA.predictor.feature_importance(dATA.test)
    )

def load_new(target, path):
    try:
        dATA.predictor = TabularPredictor.load(path)
        t = dATA.test.drop(columns=[target])
        preds = dATA.predictor.predict(t)
        print(dATA.predictor.feature_importance(dATA.test))
        return (
            np.nan,
            'No Infomation For New Test Data', 
            show_results(None, preds),
            dATA.predictor.leaderboard(dATA.test),
            dATA.predictor.feature_importance(dATA.test)
        )
    except Exception as e:
        print(e)

def rmse(y_hat, y):
    # we already used log prices before, so we only need to compute RMSE
    return sum((y_hat - y)**2 / len(y))**0.5

def show_results(y_test, y_pred):
    plt.clf()
    plt.figure(figsize=(8, 6))
    if y_test is not None:
        plt.scatter(y_test, y_pred, alpha=0.5)
    else:
        plt.bar(range(len(y_pred)), y_pred) # 将y_pred作为y_test传入
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    plt.title('Predicted Value VS Actual Value')
    if y_test is not None:
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--') # 添加y=x参考线
    plt.savefig('./results/result.png')
    return plt

def dict_to_long_string(results_dict:dict):
    """
    将评估结果字典转换为长字符串格式。
    参数:
    results_dict (dict): 从模型评估函数返回的字典，包含各种性能指标。

    返回:
    str: 包含所有键值对的长字符串。
    """
    # 创建一个空字符串
    long_string = "评估结果：\n"
    # 遍历字典，添加每个项到字符串
    for key, value in results_dict.items():
        long_string += f"{key}: {value:.4f}\n"

    return long_string

def merge_new(path:str):
    if path:
        if path.endswith('.csv'):
            t = pd.read_csv(path)
        elif path.endswith('.xlsx'):
            t = pd.read_excel(path)
        else:
            t = pd.read_feather(path)
    else:
        return dATA.info, dATA.find_string
    t['Date'] = '2021-2-15'  # 为测试数据集添加日期列

    # 找出训练集有而测试集没有的列
    missing_columns = set(dATA.data.columns) - set(t.columns)
    # 在测试集中为这些缺失的列添加NaN值
    for column in missing_columns:
        t[column] = np.nan  # 为测试数据集的预测列赋值 NaN
    if dATA.data is not None:
        dATA.data['Date'] = '2020-1-1'
        dATA.data = pd.concat([dATA.data, t], axis=0, ignore_index=True, sort=False)

    return dATA.info, dATA.find_string