import ast
import numpy as np
import matplotlib.pyplot as plt

def read_jsonl(file_path):
    import json
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# 读取数据
data = read_jsonl(r'/data/sunwh/idrr/results/clf/roberta_ft/epo5_cosine/log.jsonl')

# 提取三个指标
f1_scores = []
acc_scores = []
haullation_scores = []

for it in data[:-1]:
    if 'eval_Acc' in it.keys():
        f1_scores.append(it['eval_Macro-F1'])
        acc_scores.append(it['eval_Acc'])
        confusion_matrix = np.array(it['eval_Confusion_Matrix_Normalized'])
        eval_haullation = (confusion_matrix[0, 2] + confusion_matrix[1, 2] + confusion_matrix[3, 2]) / 3    
        haullation_scores.append(eval_haullation)
        print('{:.4f}\t{:.4f}\t{:.4f}'.format(it['eval_Macro-F1'], it['eval_Acc'], eval_haullation))

# 创建折线图
plt.figure(figsize=(12, 8))

# 绘制三条折线
epochs = [i*(158/790) for i in range(1, len(f1_scores) + 1)]
plt.plot(epochs, f1_scores, 'b-o', label='Macro F1 Score', linewidth=2, markersize=6)
plt.plot(epochs, acc_scores, 'r-s', label='Accuracy', linewidth=2, markersize=6)
plt.plot(epochs, haullation_scores, 'g-^', label='Hallucination Score', linewidth=2, markersize=6)

# 设置图表属性
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Training Metrics Over Time', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# 设置y轴范围，使图表更清晰
plt.ylim(0, 1)

# 添加数值标注（如果数据点不多的话）
if len(f1_scores) <= 20:
    for i, (f1, acc, hall) in enumerate(zip(f1_scores, acc_scores, haullation_scores)):
        if i % 2 == 0:  # 每隔一个点标注，避免过于拥挤
            plt.annotate(f'{f1:.3f}', (i+1, f1), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, alpha=0.7)
            plt.annotate(f'{acc:.3f}', (i+1, acc), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, alpha=0.7)
            plt.annotate(f'{hall:.3f}', (i+1, hall), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8, alpha=0.7)

plt.tight_layout()

# 保存图片
plt.savefig('/data/sunwh/training_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印统计信息
print(f"\n统计信息:")
print(f"F1 Score - 最高: {max(f1_scores):.4f}, 最低: {min(f1_scores):.4f}, 平均: {np.mean(f1_scores):.4f}")
print(f"Accuracy - 最高: {max(acc_scores):.4f}, 最低: {min(acc_scores):.4f}, 平均: {np.mean(acc_scores):.4f}")
print(f"Hallucination - 最高: {max(haullation_scores):.4f}, 最低: {min(haullation_scores):.4f}, 平均: {np.mean(haullation_scores):.4f}")

print(f"\n折线图已保存到: /data/sunwh/training_metrics.png") 