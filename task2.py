import plotly.graph_objects as go
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# ==========================================
# 1. 准备数据
# ==========================================
# 加载鸢尾花数据集
iris = load_iris()
# 选择前三个特征：[花萼长度, 花萼宽度, 花瓣长度]
X = iris.data[:, :3]
y = iris.target

# ==========================================
# 2. 训练模型
# ==========================================
# 使用逻辑回归模型，设置为多分类模式
clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200, random_state=42)
clf.fit(X, y)
# 获取模型系数和截距
W = clf.coef_
b = clf.intercept_

# ==========================================
# 3. 创建网格
# ==========================================
# (Plotly 处理能力强，步长可以设小一点，比如 0.1)
# 稍微扩大网格范围，以免边界在边缘被截断
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# ==========================================
# 4. 辅助函数：计算边界并裁剪
# ==========================================
def get_clipped_boundary(c1, c2, xx, yy):
    """
    计算两个类别 c1 和 c2 之间的决策边界平面，并进行裁剪。
    """
    # 决策边界方程: (W_c1 - W_c2) * x + (b_c1 - b_c2) = 0
    w_diff = W[c1] - W[c2]
    b_diff = b[c1] - b[c2]
    
    # 如果 z 的系数接近 0，说明平面垂直于 z 轴，无法用 z = f(x, y) 表示，这里简单跳过
    if np.abs(w_diff[2]) < 1e-5: return None
    
    # 计算 Z 平面坐标
    zz = -(w_diff[0] * xx + w_diff[1] * yy + b_diff) / w_diff[2]
    
    # 裁剪逻辑 (Masking)
    # 只有当该边界是当前点的最优决策边界时才显示
    grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    scores = grid_points.dot(W.T) + b
    pred = np.argmax(scores, axis=1).reshape(xx.shape)
    
    # 将不属于 c1 或 c2 区域的点设为 None (Plotly 中用 None 表示不画)
    zz_masked = np.where((pred == c1) | (pred == c2), zz, np.nan)
    
    # 限制 Z 轴范围，太高太低的不要 (稍微放宽范围)
    z_limit = (zz_masked >= X[:, 2].min() - 1.0) & (zz_masked <= X[:, 2].max() + 1.0)
    zz_masked = np.where(z_limit, zz_masked, np.nan)
    
    return zz_masked

# ==========================================
# 5. 开始绘图
# ==========================================
fig = go.Figure()

# --- 绘制三个决策平面 ---
# 两两组合类别：(0, 1), (1, 2), (2, 0)
pairs = [(0, 1), (1, 2), (2, 0)]
colors = ['purple', 'cyan', 'orange']
labels = ['Boundary: Setosa vs Versicolor', 'Boundary: Versicolor vs Virginica', 'Boundary: Virginica vs Setosa']

for (c1, c2), color, label in zip(pairs, colors, labels):
    zz = get_clipped_boundary(c1, c2, xx, yy)
    if zz is not None:
        fig.add_trace(go.Surface(
            x=xx, y=yy, z=zz,
            colorscale=[[0, color], [1, color]], # 单色显示
            showscale=False,
            opacity=0.5,
            name=label,
            showlegend=True, # 强制显示图例
            hoverinfo='name' # 鼠标悬停显示平面名称
        ))

# --- 绘制散点数据 ---
scatter_colors = ['red', 'green', 'blue']
species = ['Setosa', 'Versicolor', 'Virginica']

for i in range(3):
    fig.add_trace(go.Scatter3d(
        x=X[y==i, 0], y=X[y==i, 1], z=X[y==i, 2],
        mode='markers',
        # 恢复较小的点大小 (size=2)，图例大小由 layout 中的 itemsizing 控制
        marker=dict(size=2, color=scatter_colors[i], opacity=0.9),
        name=species[i]
    ))

# ==========================================
# 6. 设置布局
# ==========================================
fig.update_layout(
    # --- 修改部分开始 ---
    title={
        'text': "Interactive 3D Decision Boundaries (Plotly)",
        'y': 0.9,           # 垂直位置（靠近顶部，范围0-1）
        'x': 0.38,            
        'xanchor': 'center', # 锚点：让标题文本的中心点对齐到 x=0.5 的位置
        'yanchor': 'top'     # 垂直对齐方式
    },
    # --- 修改部分结束 ---
    
    legend=dict(itemsizing='constant'), 
    scene=dict(
        xaxis_title='Sepal Length',
        yaxis_title='Sepal Width',
        zaxis_title='Petal Length',
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=0.8)
    ),
    margin=dict(l=0, r=0, b=0, t=50) # 注意：稍微增加一点 t (top margin)，防止标题被切掉或太挤
)

fig.show()
fig.write_html("iris_3d_boundary.html")