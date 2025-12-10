import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 数据准备与模型训练
# ==========================================
# 加载 Iris 数据集
iris = load_iris()
X_raw = iris.data
y_raw = iris.target

# 筛选数据：只保留 Setosa (0) 和 Virginica (2) 两类
# y_raw != 1 意味着排除了 Versicolor (1)
mask = y_raw != 1
X_filtered = X_raw[mask]
y_filtered = y_raw[mask]

# 特征选择：选取第0列 (Sepal Length) 和 第2列 (Petal Length)
X_data = X_filtered[:, [0, 2]]
y_data = y_filtered

# SVM (特别是 RBF 核) 对特征的尺度非常敏感，必须标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_data)

# --- 使用 SVM (RBF核) ---
# probability=True: 允许输出概率
model = SVC(kernel='rbf', C=1.0, gamma=0.5, probability=True)
model.fit(X_scaled, y_data)

# ==========================================
# 2. 生成网格数据 (用于 3D 绘图)
# ==========================================
# 动态计算数据的范围 (外扩 1 个单位，留出边缘空间)
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1

# 生成网格坐标矩阵
xx = np.linspace(x_min, x_max, 100)
yy = np.linspace(y_min, y_max, 100)
X, Y = np.meshgrid(xx, yy)

# 计算网格中每个点的预测概率
# predict_proba 返回 (N, 2) 的数组
probs = model.predict_proba(np.c_[X.ravel(), Y.ravel()])

# 计算概率差值：P(Virginica) - P(Setosa)
# 结果范围在 -1 到 1 之间
Z = probs[:, 1] - probs[:, 0]
Z = Z.reshape(X.shape)

# ==========================================
# 3. 绘图设置 
# ==========================================

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 绘制 3D 曲面
surf = ax.plot_surface(X, Y, Z, 
                       color='#A0C8F0',       # 柔和的浅蓝色填充 
                       edgecolor='#1E56A0',   # 鲜艳的皇家蓝网格线 (对比度高)
                       linewidth=0.8,         # 线条加粗一点
                       antialiased=True,      
                       rstride=5, cstride=5,  # 网格稍微稀疏一点
                       alpha=0.4)             # 增加透明度，让后面的网格能透出来             

# --- B. 绘制投影等高线 (Contour Map) ---
# 在 3D 盒子的底部和侧壁绘制投影，增强立体感
z_bottom = -1.5 # 底部投影位置
z_top = 1.5

# zdir指定投影方向，offset指定投影平面的位置
ax.contourf(X, Y, Z, zdir='z', offset=z_bottom, cmap='coolwarm', alpha=0.7) # 底部投影
ax.contourf(X, Y, Z, zdir='y', offset=y_max+1, cmap='coolwarm', alpha=0.7)  # 侧壁投影
ax.contourf(X, Y, Z, zdir='x', offset=x_min-1, cmap='coolwarm', alpha=0.7)  # 后壁投影

# ==========================================
# 4. 坐标轴与美化
# ==========================================

# 锁定坐标轴范围，确保投影显示在视野边缘
ax.set_zlim(z_bottom, z_top)
ax.set_xlim(x_min-1, x_max+1)
ax.set_ylim(y_min-1, y_max+1)

# 设置坐标轴标签
ax.set_xlabel('Sepal Length (Standardized)')
ax.set_ylabel('Petal Length (Standardized)')
ax.set_zlabel('P(Virginica) - P(Setosa)')

# 设置初始观察视角 (高度角 elev, 方位角 azim)
ax.view_init(elev=30, azim=-60)

plt.title('SVM (RBF Kernel) Probability Surface')
plt.show()