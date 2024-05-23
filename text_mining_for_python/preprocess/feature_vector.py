import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# this function can translate raw data to feature vector
def get_feature(text_data):
    # 将text分割成sentence
    sentences = text_data.split('.')

    # 创建CountVectorizer对象，并进行特征提取
    # count_vectorizer = CountVectorizer()
    # count_features = count_vectorizer.fit_transform(sentences)

    # 使用TfidfVectorizer进行特征提取
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_features = tfidf_vectorizer.fit_transform(sentences)

    # 计算每个特征的TF-IDF值之和
    feature_tfidf_sum = tfidf_features.sum(axis=0)

    # 创建特征词汇-TF-IDF值字典
    feature_tfidf_dict = {
        'feature': tfidf_vectorizer.get_feature_names_out(),
        'tfidf_sum': feature_tfidf_sum.tolist()[0]
    }

    # 将字典转换为DataFrame，并根据TF-IDF值降序排序
    df = pd.DataFrame(feature_tfidf_dict).sort_values(by='tfidf_sum', ascending=False)
    df = df.reset_index(drop=True)

    # 打印排序后的特征词汇及其对应的TF-IDF值 if is needed
    # for index, row in df.iterrows():
    #     print(row['feature'], row['tfidf_sum'])
    print(df)
    return df[0: 20]


def get_feature_expamle():
    text_data = ("hood associate judge appeal conviction offense commonly known disorderly conduct statute part make "
                 "unlawful person use profane language indecent obscene word engage disorderly conduct street avenue "
                 "public place evidence effect complaining witness escort member armed service engaged taxicab "
                 "operated appellant take union station station witness escort left cab instructed appellant drive "
                 "home several mile away northwest section district way appellant made certain suggestion remark "
                 "witness indecent obscene nature repeated suggestion en route upon reaching destination occurred "
                 "clock morning appellant admitted driver taxicab denied making remark trial without jury court found "
                 "appellant guilty appellant chief contention taxicab private place time remark occupied cab "
                 "considered public place within meaning statute therefore offense committed undoubtedly statute "
                 "directed conduct public place statute commonly prohibit use public place word lewd obscene profane "
                 "insulting fighting word spoken face face likely incite immediate breach peace appellant made remark "
                 "complaining witness standing street preparing enter cab would clear violation statute fact made "
                 "remark cab make difference think taxicab common carrier public utility deriving income use public "
                 "street avenue subject call member public often occupied one passenger one group common knowledge "
                 "today vehicle may frequently carry number wholly unrelated unacquainted person fact complaining "
                 "witness passenger time defense presence others offender person addressed necessary complete offense "
                 "satisfied public vehicle plying business public street public place within meaning statute "
                 "appellant also contends remark constitute profane language indecent obscene word record show use "
                 "profanity appellant without detailing appellant remark think ample justification trial judge "
                 "finding remark indecent obscene word indecent obscene susceptible exact definition determining "
                 "whether remark appellant within prohibition statute trial judge entitled consider surrounding "
                 "circumstance time occurrence manner occurred repetition remark well lack previous acquaintance "
                 "finally appellant complains information also charged attempted engage conversation certain female")

    df = get_feature(text_data)
    print("this is feature data example: \nbelow is (feature) and (weight)")
    # 以表格形式打印DataFrame
    print(df)

    # this is print data:

    # this is feature data example:
    # below is (feature) and (weight)
    #         feature  tfidf_sum
    # 0     appellant   0.396925
    # 1        remark   0.330771
    # 2        public   0.330771
    # 3       statute   0.231539
    # 4       obscene   0.198462
    # ..          ...        ...
    # 169     finding   0.033077
    # 170       found   0.033077
    # 171  frequently   0.033077
    # 172       group   0.033077
    # 173       would   0.033077
    #
    # [174 rows x 2 columns]
