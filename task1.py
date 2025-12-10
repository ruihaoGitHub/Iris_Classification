import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from sklearn import datasets
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    PolynomialFeatures,
    SplineTransformer,
)

# ==========================================
# 1. 数据加载与预处理
# ==========================================
# 加载鸢尾花数据集
iris = datasets.load_iris()
# 仅使用前两个特征（花萼长度和花萼宽度）以便于二维可视化
X = iris.data[:, 0:2]
y = iris.target

# 将数据集划分为训练集和测试集，测试集占比30%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ==========================================
# 2. 定义分类器
# ==========================================
# 定义包含不同特征预处理方法的分类器字典
classifiers = {
    # 逻辑回归（使用 Nystroem 近似 RBF 核特征）
    "Logistic regression\n(RBF features)": make_pipeline(
        Nystroem(kernel="rbf", gamma=5e-1, n_components=50, random_state=1),
        LogisticRegression(C=10),
    ),
    # 梯度提升分类器（原生支持非线性）
    "Gradient\nBoosting": HistGradientBoostingClassifier(),
    # 逻辑回归（使用分箱特征 + 多项式特征）
    "Logistic regression\n(binned features)": make_pipeline(
        KBinsDiscretizer(n_bins=5, quantile_method="averaged_inverted_cdf"),
        PolynomialFeatures(interaction_only=True),
        LogisticRegression(C=10),
    ),
    # 逻辑回归（使用样条特征 + 多项式特征）
    "Logistic regression\n(spline features)": make_pipeline(
        SplineTransformer(n_knots=5),
        PolynomialFeatures(interaction_only=True),
        LogisticRegression(C=10),
    ),
}

# ==========================================
# 3. 绘图设置
# ==========================================
n_classifiers = len(classifiers)
# 散点图的样式参数
scatter_kwargs = {
    "s": 25,
    "marker": "o",
    "linewidths": 0.8,
    "edgecolor": "k",
    "alpha": 0.7,
}
y_unique = np.unique(y)

# 确保图例不被截断
mpl.rcParams["savefig.bbox"] = "tight"
# 创建子图网格：行数为分类器数量，列数为类别数量+1（最后一列为最大概率类别）
fig, axes = plt.subplots(
    nrows=n_classifiers,
    ncols=len(iris.target_names) + 1,
    figsize=(4 * 2, n_classifiers * 9 + 2)
)
levels = 100 # 等高线层级数

# ==========================================
# 4. 训练模型并绘制决策边界
# ==========================================
for classifier_idx, (name, classifier) in enumerate(classifiers.items()):
    # 训练模型并进行预测
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    
    # 遍历每个类别，绘制该类别的概率分布图
    for label in y_unique:
        # 绘制分类器提供的概率估计（决策边界）
        disp = DecisionBoundaryDisplay.from_estimator(
            classifier,
            X_train,
            response_method="predict_proba",
            class_of_interest=label,
            ax=axes[classifier_idx, label],
            vmin=0,
            vmax=1,
            cmap="Blues",
            levels=levels,
        )
        axes[classifier_idx, label].set_title(f"Class {label}")
        
        # 绘制预测属于该类别的数据点
        mask_y_pred = y_pred == label
        axes[classifier_idx, label].scatter(
            X_test[mask_y_pred, 0], X_test[mask_y_pred, 1], c="w", **scatter_kwargs
        )

        axes[classifier_idx, label].set(xticks=(), yticks=())
    
    # 添加一列显示所有类别，通过绘制具有最大 'predict_proba' 的类别
    max_class_disp = DecisionBoundaryDisplay.from_estimator(
        classifier,
        X_train,
        response_method="predict_proba",
        class_of_interest=None,
        ax=axes[classifier_idx, len(y_unique)],
        vmin=0,
        vmax=1,
        levels=levels,
    )
    
    # 在最大概率列中绘制真实标签的数据点
    for label in y_unique:
        mask_label = y_test == label
        axes[classifier_idx, 3].scatter(
            X_test[mask_label, 0],
            X_test[mask_label, 1],
            c=max_class_disp.multiclass_colors_[[label], :],
            **scatter_kwargs,
        )

    axes[classifier_idx, 3].set(xticks=(), yticks=())
    axes[classifier_idx, 3].set_title("Max class")
    axes[classifier_idx, 0].set_ylabel(name, fontsize=6)

# ==========================================
# 5. 调整布局与添加颜色条
# ==========================================
# 调整布局以防止重叠，并为底部 colorbar 留出空间
plt.subplots_adjust(wspace=0.1, hspace=0.39, bottom=0.3)

# 为单个类别的概率图添加颜色条
ax_single = fig.add_axes([0.15, 0.1, 0.5, 0.02])
plt.title("Probability")
_ = plt.colorbar(
    cm.ScalarMappable(norm=None, cmap=disp.surface_.cmap),
    cax=ax_single,
    orientation="horizontal",
)

# 为最大概率类别列添加颜色条
max_class_cmaps = [s.cmap for s in max_class_disp.surface_]

for label in y_unique:
    ax_max = fig.add_axes([0.73, (0.20 - (label * 0.07)), 0.16, 0.015])
    plt.title(f"Probability class {label}", fontsize=10)
    _ = plt.colorbar(
        cm.ScalarMappable(norm=None, cmap=max_class_cmaps[label]),
        cax=ax_max,
        orientation="horizontal",
    )
    if label in (0, 1):
        ax_max.set(xticks=(), yticks=())

plt.show()