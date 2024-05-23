import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import plot_tree
from sklearn.tree import export_text
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def random_forest():
    df = pd.read_csv('../data_featured.csv', encoding='latin1')

    # 移除非特征列
    features = df.drop(['id', 'name', 'decision_date', 'type'], axis=1)
    labels = df['type']  # 标签列是 'type'

    # 分割数据为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # 创建随机森林分类器
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # n_estimators 表示决策树的数量
    rf_model.fit(X_train, y_train)  # 训练模型

    # 预测测试集
    y_pred_rf = rf_model.predict(X_test)

    # 输出模型的分类报告和准确率
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf))
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

    # 生成混淆矩阵
    conf_mat = confusion_matrix(y_test, y_pred_rf)

    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix for random forest')
    plt.show()


    # 选择随机森林中的一棵树
    tree = rf_model.estimators_[0]
    tree_structure = export_text(tree, feature_names=list(features.columns))
    print("Structure of one Decision Tree:")
    print(tree_structure)

    # 绘制一棵决策树的图形表示
    plt.figure(figsize=(20, 10))
    plot_tree(tree, feature_names=features.columns, class_names=['Class 0', 'Class 1'], filled=True, impurity=True,
              fontsize=10)
    plt.title('Visualization of a Decision Tree in RandomForest')
    plt.show()


random_forest()