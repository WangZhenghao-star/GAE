import torch
from recode.utils import output_adj
from recode.gae import GAE
import pandas as pd
import numpy as np
from recode.causal_strength import adj_cs

# 加载数据集，确保第一行被识别为列名
X = pd.read_csv("../data/dataset/2011-2020_1.csv")

# 检查列名，确认没有意外的字符串
print(X.columns)

# 如果'Outcome'列存在，并且您想要删除它
if 'CLASS' in X.columns:
    X = X.drop(columns=['CLASS'])
else:
    print("列 'CLASS' 不存在。")

# 将DataFrame转换为NumPy数组，并确保数据类型为float32
X_np = X.values.astype(np.float32)  # 将数据转换为NumPy数组

# 初始化GAE模型
gae = GAE(epochs=10, device_type="cpu")

# 训练模型
gae.learn(X_np)  # 直接传递NumPy数组

# 输出因果关系邻接矩阵
print(gae.causal_matrix)

# 输出因果强度邻接矩阵（如果需要）
print(adj_cs(gae.causal_matrix, X_np))

# 使用recode.utils中的output_adj函数输出邻接矩阵
output_adj(gae.causal_matrix)
