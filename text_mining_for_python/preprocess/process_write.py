import pandas as pd


def process_and_write(data_list):
    # 创建全局词汇表
    all_words = set()
    # 元数据字段列表
    metadata_fields = ['id', 'name', 'decision_date', 'type']
    for data in data_list:
        # 过滤出不是元数据字段的词汇
        filtered_features = [word for word in data['options']['feature'] if word not in metadata_fields]
        all_words.update(filtered_features)
    all_words = list(all_words)

    # 准备合并的数据
    rows = []
    for data in data_list:
        # 每个词的权重字典
        feature_weights = dict(zip(data['options']['feature'], data['options']['tfidf_sum']))
        # 创建行字典，初始化所有词的权重为0，确保不包括元数据字段
        row = dict.fromkeys(all_words, 0)
        row.update(feature_weights)  # 更新当前数据集中有的词的权重
        # 添加元数据字段
        for field in metadata_fields:
            row[field] = data[field]
        # 添加到行列表中
        rows.append(row)

    # 创建 DataFrame
    df = pd.DataFrame(rows)
    # 直接删除非零值数量小于8的列
    df = df.loc[:, (df != 0).sum() >= 15]
    print(df)
    df.to_csv('data_featured.csv', index=False)
