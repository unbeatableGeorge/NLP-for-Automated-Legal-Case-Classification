########################################################################################
#                                                                                      #
#        auth: Yifan Xiang  (向奕帆)                                                    #
#        module: data mining                                                           #
#        leeds id: 201587306                                                           #
#        SWJTU id: 2021110026                                                          #
#                                                                                      #
########################################################################################

# -------------------------------------------------->
# |
# |-- dataset
# |   |-- civil
# |   |-- criminal
# |   |-- civ_case.csv
# |   |-- civ_case.csv
# |   `-- preprocessing.py
# |
# |-- img
# |
# |-- main.py                         main read file from data set and get data
# |                                                       |
# |-- text_cleaner.py               text_data is pre-processed data generate from text
# |                                                       |
# |-- feature_vector.py                        word_weight is df type
# |                                                       |
# |-- img_drawer.py                                draw if is needed
# |                                                       |
# |-- process_write.py                          process and write in csv
# |
# `-- README.md
# -------------------------------------------------->


import pandas as pd
from text_mining_for_python.preprocess.feature_vector import get_feature
from text_mining_for_python.preprocess.img_drawer import img_drawer
from text_mining_for_python.preprocess.process_write import process_and_write
from text_mining_for_python.preprocess.text_cleaner import text_cleaner


exceptions = [346, 386, 388, 406, 407, 440]  # some exception index which will not be processed


def read_data_from_csv():
    df_civ = pd.read_csv('./dataset/civ_case.csv', encoding='latin1')
    df_cri = pd.read_csv('./dataset/crim_case.csv', encoding='latin1')
    df_civ['type'] = 0  # civil case
    df_cri['type'] = 1  # criminal case
    # 使用 concat 垂直堆叠
    result = pd.concat([df_civ, df_cri], ignore_index=True)
    return result


if __name__ == '__main__':
    data = read_data_from_csv()
    data_list = []
    for index, row in data.iterrows():
        if index in exceptions:
            continue
        print("processing the " + str(index) + "th row")
        print("    name:" + str(row['name']))
        text_data = text_cleaner(row['options'])            # text_data is pre-processed data generate from text
        words_weight = get_feature(text_data)               # word_weight is df type
        img_drawer(row['id'], words_weight)               # draw if is needed
        data_list.append({
            'id': row['id'],
            'name': row['name'],
            'decision_date': row["decision_date"],
            'type': row["type"],
            'options': words_weight
        })
    process_and_write(data_list)

    # 支持向量机 SVM


