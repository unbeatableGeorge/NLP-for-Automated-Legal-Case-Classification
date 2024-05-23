import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report
import torch

# 加载数据
df = pd.read_csv('../dataset/cases.xlsx', encoding='latin1')

# 假设文本数据存储在'text'列，'type'是标签
texts = df['text']  # 需要根据实际文本列名称调整
labels = df['type']

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 加载预训练的BERT模型和分词器
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(labels.unique()))

# 准备输入数据
def encode_texts(tokenizer, texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')

train_encodings = encode_texts(tokenizer, X_train.tolist())
test_encodings = encode_texts(tokenizer, X_test.tolist())

# 转换为torch数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_encodings, y_train.tolist())
test_dataset = Dataset(test_encodings, y_test.tolist())

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',          # 输出目录
    num_train_epochs=3,              # 训练轮次
    per_device_train_batch_size=8,   # 训练批大小
    per_device_eval_batch_size=16,   # 测试批大小
    warmup_steps=500,                # 预热步骤
    weight_decay=0.01,               # 权重衰减
    logging_dir='./logs',            # 日志目录
    logging_steps=10,
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# 训练模型
trainer.train()

# 预测
predictions = trainer.predict(test_dataset)
preds = torch.argmax(predictions.predictions, dim=-1)

# 计算准确率和分类报告
print("Accuracy:", accuracy_score(y_test, preds.numpy()))
print("Classification Report:")
print(classification_report(y_test, preds.numpy()))
