import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def svm_model():
    df = pd.read_csv('../data_featured.csv', encoding='latin1')

    # 假设df是你的DataFrame，其中包括所有词的权重和type列
    features = df.drop(['id', 'name', 'decision_date', 'type'], axis=1)  # 去除非特征列
    labels = df['type']  # 目标列是type

    # 分割数据为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    # 创建SVM分类器
    model = svm.SVC(kernel='linear')  # 可以选择 'linear', 'poly', 'rbf' 等核函数
    model.fit(X_train, y_train)  # 训练模型
    # 预测测试集
    y_pred = model.predict(X_test)

    # 输出模型的分类报告和准确率
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # 生成混淆矩阵
    conf_mat = confusion_matrix(y_test, y_pred)

    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


svm_model()
