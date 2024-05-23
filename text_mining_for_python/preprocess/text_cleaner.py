# this is text_cleaner for text which can get raw data (清洗后的数据 下面有 example)
# and there is a example below
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def text_cleaner(text):

    # Remove special characters and digits
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r'\d+', '', text)

    # Convert to lowercase
    text = text.lower()

    # Tokenize the text into individual words
    words = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Print the cleaned text
    cleaned_text = ' '.join(words)

    return cleaned_text


def text_cleaner_example():
    text = ("HOOD, Associate Judge.\nThis is an appeal from a conviction of the offense commonly known as disorderly "
            "conduct. The statute, in part, makes it unlawful for any person to use profane language or indecent or "
            "obscene words, or engage in any disorderly conduct in any street, avenue, or other public place.\nThe "
            "evidence was to the effect that the complaining witness and her escort, a member of the armed services, "
            "engaged a taxicab operated by the appellant to take them to the Union Station, that at the station the "
            "witness\u2019 escort left the cab and she instructed appellant to drive her to her home several miles "
            "away in the Northwest section of the District; that on the way appellant made certain suggestions and "
            "remarks to the witness of an indecent and obscene nature, and repeated the suggestions both en route and "
            "upon reaching their destination. This occurred at about 1:30 o\u2019clock in the morning.\nAppellant "
            "admitted he was the driver of the taxicab but denied making the remarks. The trial was without a jury "
            "and the court found appellant guilty.\nAppellant\u2019s chief contention is that the taxicab was a "
            "private place at the time of the remarks, that an occupied cab at 1:30 A. M. cannot be considered a "
            "public place within the meaning of the statute, and, therefore, no offense was committed. Undoubtedly, "
            "the statute is directed at conduct in public places. Such statutes commonly prohibit the use in public "
            "places of words which are lewd, obscene or profane, and insulting or \u201cfighting\u201d words, "
            "which when spoken face to face are likely to incite an immediate breach of the peace.\nHad appellant "
            "made the remarks to the complaining witness while she was standing in the street and preparing to enter "
            "the cab, there would have been a clear violation of the statute. Does the fact he made the remarks after "
            "she was in the cab make a difference? We think not.\nA taxicab is a common carrier and public utility, "
            "deriving its income from the use of public streets and avenues, subject to the call of any member of the "
            "public, and while often occupied by only one passenger or one group, it is common knowledge that today "
            "such vehicles may and frequently do carry a number of wholly unrelated and unacquainted persons. The "
            "fact that the complaining witness was the only passenger at the time is no defense. The presence of "
            "others than the offender and the person addressed is not necessary to complete the offense.\nWe are "
            "satisfied that a public vehicle plying its business on a public street is a public place within the "
            "meaning of the statute.\nAppellant also contends that his remarks did not constitute \u201cprofane "
            "language, indecent and obscene words.\u201d The record shows no use of profanity by appellant, but, "
            "without detailing appellant\u2019s remarks, we think there was ample justification for the trial judge "
            "finding such remarks indecent and obscene. The words \u201cindecent\u201d and \u201cobscene\u201d are "
            "not susceptible of exact definition, and in determining whether the remarks of appellant were within the "
            "prohibition of the statute, the trial judge was entitled to consider all the surrounding circumstances, "
            "the time of the occurrence and the manner in which it occurred, the repetition of those remarks, "
            "as well as the lack of previous acquaintance.\nFinally, appellant complains the information also charged "
            "that \u201che attempted to engage in conversation a certain female")

    print("This is printed test data:")
    print(text_cleaner(text))

    # for this text the print is:
    # ("hood associate judge appeal conviction offense commonly known disorderly conduct statute part make unlawful "
    #  "person use profane language indecent obscene word engage disorderly conduct street avenue public place "
    #  "evidence effect complaining witness escort member armed service engaged taxicab operated appellant take "
    #  "union station station witness escort left cab instructed appellant drive home several mile away northwest "
    #  "section district way appellant made certain suggestion remark witness indecent obscene nature repeated "
    #  "suggestion en route upon reaching destination occurred clock morning appellant admitted driver taxicab "
    #  "denied making remark trial without jury court found appellant guilty appellant chief  contention taxicab "
    #  "private place time remark occupied cab considered public place within meaning statute therefore offense "
    #  "committed undoubtedly statute directed conduct public place statute commonly prohibit use public place word "
    #  "lewd obscene profane insulting fighting word spoken face face likely incite immediate breach peace "
    #  "appellant made remark complaining witness standing street preparing enter cab would clear violation statute "
    #  "fact made remark cab make difference think taxicab common carrier public utility deriving income use public "
    #  "street avenue subject call member public often occupied one passenger one group common knowledge today "
    #  "vehicle may frequently carry number wholly unrelated unacquainted person fact complaining witness passenger "
    #  "time defense presence others offender person addressed necessary complete offense satisfied public vehicle "
    #  "plying business public street public place within meaning statute appellant also contends remark constitute "
    #  "profane language indecent obscene word record show use profanity appellant without detailing appellant "
    #  "remark think ample justification trial judge finding remark indecent obscene word indecent obscene "
    #  "susceptible exact definition determining whether remark appellant within prohibition statute trial judge "
    #  "entitled consider surrounding circumstance time occurrence manner occurred repetition remark well lack "
    #  "previous acquaintance finally appellant complains information also charged attempted engage conversation "
    #  "certain female")
