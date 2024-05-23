import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud


def img_drawer(name, df):
    text = []
    for index, row in df.iterrows():
        if index > 40:
            break
        text.append(row['feature'])
    text = ' '.join(text)

    # open bg cover
    img = Image.open('./img/img_cover/cover.jpg')
    img_array = np.array(img)

    # 创建词云对象
    wc = WordCloud(
        background_color='white',
        width=800,
        height=800,
        mask=img_array,
    )

    # 生成词云图
    wc.generate_from_text(text)

    # 显示词云图
    plt.imshow(wc)
    plt.axis('off')
    plt.show()

    # 保存词云图
    wc.to_file("./img/img_generate/" + str(name) + ".jpg")
