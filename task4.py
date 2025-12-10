import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

from sklearn import datasets
from sklearn.ensemble import HistGradientBoostingClassifier
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
# 1. 数据准备 (升级为3个特征)
# ==========================================
# 加载鸢尾花数据集
iris = datasets.load_iris()
# 特征选择: 0:Sepal Length, 1:Sepal Width, 2:Petal Length
feature_indices = [0, 1, 2] 
feature_names = [iris.feature_names[i] for i in feature_indices]
X = iris.data[:, feature_indices]
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ==========================================
# 2. 模型定义
# ==========================================
# 定义包含不同特征预处理方法的分类器字典
classifiers = {
    # 逻辑回归（使用 Nystroem 近似 RBF 核特征）
    "Logistic Regression\n(RBF features)": make_pipeline(
        Nystroem(kernel="rbf", gamma=5e-1, n_components=50, random_state=1),
        LogisticRegression(C=10),
    ),
    # 梯度提升分类器
    "Gradient Boosting": HistGradientBoostingClassifier(),
    # 逻辑回归（使用分箱特征 + 多项式特征）
    "Logistic Regression\n(Binned features)": make_pipeline(
        KBinsDiscretizer(n_bins=5, quantile_method="averaged_inverted_cdf"),
        PolynomialFeatures(interaction_only=True),
        LogisticRegression(C=10),
    ),
    # 逻辑回归（使用样条特征 + 多项式特征）
    "Logistic Regression\n(Spline features)": make_pipeline(
        SplineTransformer(n_knots=5),
        PolynomialFeatures(interaction_only=True),
        LogisticRegression(C=10),
    ),
}

# 训练所有模型
print("正在训练模型...")
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)

# ==========================================
# 3. 动画准备：网格生成与布局
# ==========================================
# 定义前两个维度(x, y)的网格范围 (固定不动)
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

# 定义第三个维度(z)的变化序列 (动画的帧)
# 我们将在 Petal Length 的最小值和最大值之间扫描
n_frames = 60
z_values = np.linspace(X[:, 2].min(), X[:, 2].max(), n_frames)

# 创建画布：2x2 布局展示4个分类器
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
axes_flat = axes.ravel()

# 自定义颜色 (Setosa, Versicolor, Virginica)
# 对应 class 0, 1, 2
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ['#FF0000', '#00FF00', '#0000FF']

# ==========================================
# 4. 核心动画更新函数
# ==========================================
def update(frame_idx):
    """
    更新每一帧的动画内容
    frame_idx: 当前帧的索引
    """
    current_z = z_values[frame_idx]
    
    # 构造当前的切面数据 (N, 3)
    # 前两列是网格，第三列全部填充为当前的 Petal Length
    mesh_input = np.c_[xx.ravel(), yy.ravel(), np.full(xx.ravel().shape, current_z)]
    
    fig.suptitle(f"Task 4: Dynamic Decision Boundary Slicing\n{feature_names[2]} = {current_z:.2f}", 
                 fontsize=16, fontweight='bold', y=0.98)

    for i, (name, clf) in enumerate(classifiers.items()):
        ax = axes_flat[i]
        ax.clear() # 清空上一帧
        
        # --- A. 绘制决策边界 ---
        # 预测当前切面的分类结果
        Z = clf.predict(mesh_input)
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8, vmin=0, vmax=2)
        
        # --- B. 绘制动态散点 (CT扫描效果) ---
        # 技巧：只显示 Petal Length 距离当前切面非常近的点 (+- 0.3)
        # 这样随着动画播放，数据点会像星星一样闪烁穿过切面
        mask_slice = np.abs(X[:, 2] - current_z) < 0.3
        
        if np.any(mask_slice):
            X_slice = X[mask_slice]
            y_slice = y[mask_slice]
            
            for class_idx, color in enumerate(cmap_bold):
                # 绘制该类别下的点
                curr_mask = y_slice == class_idx
                ax.scatter(X_slice[curr_mask, 0], X_slice[curr_mask, 1], 
                           c=color, edgecolor='k', s=60, label=iris.target_names[class_idx])

        # --- C. 美化图表 ---
        ax.set_title(name, fontsize=12)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        if i >= 2: # 只有最下面一行显示X轴标签
            ax.set_xlabel(feature_names[0])
        if i % 2 == 0: # 只有最左边一列显示Y轴标签
            ax.set_ylabel(feature_names[1])
            
    # 只在第一个图加图例，避免遮挡
    if frame_idx == 0:
        axes_flat[0].legend(loc='upper left', framealpha=0.5, fontsize=8)

# ==========================================
# 5. 生成与保存动画
# ==========================================
print(f"正在生成 {n_frames} 帧动画，请稍候...")
# interval=100 表示每帧间隔100ms
ani = FuncAnimation(fig, update, frames=n_frames, interval=100)

save_path = 'iris_3d_slicing_task4.gif'
ani.save(save_path, writer='pillow', fps=10)

print(f"完成！GIF 已保存至: {save_path}")
plt.close()