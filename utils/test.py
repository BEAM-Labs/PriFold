from predictor import Augmentation
import numpy as np


# 示例数据
seq = "AUGCU"
ct = np.array([
    [0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0]
])

# 实例化增强类
augmentation = Augmentation(select=0.5, replace=0.5)

# 运行多次并打印结果
for i in range(10):
    augmented_seq = augmentation(seq, ct)
    print(f"Run {i+1}: {augmented_seq}")