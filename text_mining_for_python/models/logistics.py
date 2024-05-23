import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


def logistics():
    df = pd.read_csv('../data_featured.csv', encoding='latin1')

    # 假设我们要预测的列名为 'type'，它是一个二元因变量
    features = df.drop(['id', 'name', 'decision_date', 'type'], axis=1)  # 去除非特征列
    labels = df['type']  # 目标列是 'type'

    # 分割数据为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # 创建逻辑回归分类器
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)  # 训练模型

    # 预测测试集
    y_pred_logistic = logistic_model.predict(X_test)

    # 输出模型的分类报告和准确率
    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, y_pred_logistic))
    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logistic))

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # 生成混淆矩阵
    conf_mat = confusion_matrix(y_test, y_pred_logistic)

    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix for Logistic Regression')
    plt.show()

logistics()
