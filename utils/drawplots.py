import matplotlib.pyplot as plt
import seaborn as sns
import random
from saver import dATA
from utils.prepocess import cal_normal


def plot_scatter(pos):
    plt.clf()
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].scatter(dATA.data[pos[0]], dATA.data[pos[1]], alpha=0.5)  # 透明度
    ax[1].scatter(dATA.data[pos[2]], dATA.data[pos[3]], alpha=0.5)

    # 标题与轴标签
    ax[0].set_title(f'{pos[0]} vs {pos[1]}')
    ax[1].set_title(f'{pos[2]} vs {pos[3]}')
    ax[0].set_xlabel(pos[0]); ax[0].set_ylabel(pos[1])
    ax[1].set_xlabel(pos[2]); ax[1].set_ylabel(pos[3])

    plt.tight_layout()  # 填充整个图像区域
    plt.savefig('./results/scatter.png')
    return plt

def plot_hist(pos):
    plt.clf()
    fig, ax = plt.subplots(1, 4, figsize=(12, 6))
    for i in range(4):
        ax[i].hist(dATA.data[pos[i]], bins=100, label=pos[i])  # 直方图
        ax[i].set_title(f'hist {pos[i]}')
        ax[i].set_xlabel('feature')
        ax[i].set_ylabel('frequency')

    plt.tight_layout()
    plt.savefig('./results/hist.png')
    return plt

def plot_boxplots(labels):
    plt.clf()
    data_lists = [dATA.data[col] for col in labels]  # 提取
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    for idx, (data, label) in enumerate(zip(data_lists, labels)):
        row = idx // 2  # 行索引
        col = idx % 2   # 列索引
        axes[row, col].boxplot(data)
        axes[row, col].set_title(f'{label}')  # 为子图标题

    # 隐藏所有x轴刻度
    for ax in axes.flat:
        ax.set_xticks([])

    plt.tight_layout()
    plt.savefig('./results/boxplot.png')
    return plt


def plot_corr():
    plt.clf()
    cols = dATA.data.select_dtypes(include=['number']).columns.tolist()
    cols = random.sample(cols, 5)
    ax = sns.heatmap(dATA.data[cols].corr(), annot=True, cmap='rainbow')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.savefig('./results/corr.png')
    return plt

def normalize(feature_names):
    X_train , X_norm = cal_normal(feature_names)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # 原始数据图
    ax[0].scatter(X_train[:,0], X_train[:,1])
    ax[0].set_xlabel(feature_names[0])
    ax[0].set_ylabel(feature_names[1])
    ax[0].set_title("Unnormalized")
    ax[0].axis('equal')
    
    # 标准化后的数据图
    ax[1].scatter(X_norm[:,0], X_norm[:,1])
    ax[1].set_xlabel(feature_names[0])
    ax[1].set_ylabel(feature_names[1])
    ax[1].set_title("Normalized")
    ax[1].axis('equal')
    
    fig.suptitle("Distribution of Features Before and After Normalization")
    plt.tight_layout()
    plt.savefig('./results/Normalize.png')
    return plt