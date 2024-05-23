import json
import os
import csv

# 读入数据 ------------------------------------>
# 文件路径
filename_civ = ['./civil/civ-data-1/json', './civil/civ-data-2/json']
filename_cri = ['./criminal/crim-data-1/json', './criminal/crim-data-2/json']

# 创建一个列表来存储所有 JSON 数据
all_json_data = []

# 遍历指定目录下的所有文件
for each in filename_civ:
    for filename in os.listdir(each):
        # 构建完整的文件路径
        file_path = os.path.join(each, filename)

        # 打开并读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as file:
            # 加载 JSON 数据并添加到列表中
            data = json.load(file)
            all_json_data.append({
                "id": data["id"],
                "decision_date": data["decision_date"],
                "name": data["name"],
                "options": data["casebody"]["opinions"][0]["text"]
            })


# 写文件 --------------------------------------->
filename1 = 'civ_case.csv'
filename2 = 'crim_case.csv'

# 使用 'w' 模式打开文件，如果文件不存在，将会被创建
with open(filename1, 'w', newline='', encoding='utf-8') as file:
    # 指定文件对象和字段名（列名）
    writer = csv.DictWriter(file, fieldnames=["id", "name", "decision_date", "options"])

    # 写入列名作为 CSV 文件的头部
    writer.writeheader()

    # 遍历数据，每次写入一行
    for row in all_json_data:
        writer.writerow(row)

print("write successfully!")
