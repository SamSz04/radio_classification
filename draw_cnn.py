import matplotlib.pyplot as plt
import matplotlib.patches as patches

# CNN1D 模型结构定义
layers = [
    ("Input", (512, 2)),
    ("Permute", (2, 512)),
    ("Conv1D(2→64)", (64, 512)),
    ("MaxPool1D", (64, 256)),
    ("Conv1D(64→128)", (128, 256)),
    ("MaxPool1D", (128, 128)),
    ("Conv1D(128→256)", (256, 128)),
    ("MaxPool1D", (256, 64)),
    ("Conv1D(256→512)", (512, 64)),
    ("AdaptiveAvgPool1D", (512, 1)),
    ("Flatten", (512,)),
    ("FC(512→256)", (256,)),
    ("FC(256→128)", (128,)),
    ("FC(128→12)", (12,))
]

# 绘图
fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, len(layers) + 1)
ax.set_ylim(0, 600)
ax.axis('off')

for i, (name, shape) in enumerate(layers):
    width = 0.8
    height = 200 * (shape[0] / 512 if isinstance(shape, tuple) else 0.5)
    y = 300 - height / 2
    rect = patches.Rectangle((i + 0.5, y), width, height, linewidth=1, edgecolor='black', facecolor='skyblue')
    ax.add_patch(rect)
    ax.text(i + 0.9, y + height + 20, name, ha='center', fontsize=9)
    ax.text(i + 0.9, y - 25, str(shape), ha='center', fontsize=8, color='gray')

plt.title("1D CNN Architecture for Modulation Classification", fontsize=14)
plt.tight_layout()
plt.show()
