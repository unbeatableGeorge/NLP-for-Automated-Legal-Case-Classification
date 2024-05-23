import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

def knn_visualization():
    df = pd.read_csv('../data_featured.csv', encoding='latin1')
    features = df.drop(['id', 'name', 'decision_date', 'type'], axis=1)
    labels = df['type']

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    accuracy_list = []
    k_values = range(1, 26)  # Testing K from 1 to 25

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_list.append(accuracy)

        if k == 5:  # Plot confusion matrix for k=5
            conf_mat = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix for k=5')
            plt.show()

            # 输出模型的分类报告和准确率
            print("KNN Classification Report:")
            print(classification_report(y_test, y_pred))
            print("KNN Accuracy:", accuracy_score(y_test, y_pred))

    # Plotting accuracy vs K value
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracy_list, marker='o', linestyle='-', color='b')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. K Value')
    plt.grid(True)
    plt.xticks(k_values)
    plt.show()

knn_visualization()
