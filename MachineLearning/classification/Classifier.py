import os
import json
from util.FileReader import read_main_info, load_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from word_segment.WordSegment import word_segment
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

JSON_FILE = "category.json"


def load_data_to_json(root_path="/Users/leimiaomiao/Desktop/行业分类",
                      extract=True,
                      decode="utf-8"):
    """
    Pre-process xml files, including extract main info from xml, and load category info.
    :param root_path:
    :param extract:
    :param decode:
    :return:
    """
    category_map = {}

    for file in os.listdir(root_path):
        # 行业分类文件夹
        path = os.path.join(root_path, file)
        if os.path.isdir(path):
            for xml_file in os.listdir(path):
                # 读取每个行业分类的文件
                xml_file_path = os.path.join(path, xml_file)
                try:
                    if xml_file_path.endswith(".xml"):
                        doc = read_main_info(xml_file_path, extract=extract, decode=decode)

                        if file in category_map.keys():
                            category_map[file].append(doc)
                        else:
                            category_map[file] = list()
                            category_map[file].append(doc)

                except UnicodeDecodeError:
                    print("UnicodeDecodeError:%s" % xml_file_path)
                    continue

    with open(JSON_FILE, "w") as f:
        json.dump(category_map, f)


def load_data_set_from_json(ratio=0.7):
    """
    load training set and testing set from json
    :param ratio:
    :return:
    """
    train_doc_list = []
    train_category_list = []

    test_doc_list = []
    test_category_list = []
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r") as f:
            category_map = json.load(f)
            categories = category_map.keys()

            for category in categories:
                all_doc_list = category_map.get(category)
                length = len(all_doc_list)
                train_set_length = int(length * ratio)

                for i in range(length):
                    if i < train_set_length:
                        train_doc_list.append(all_doc_list[i])
                        train_category_list.append(category)
                    else:
                        test_doc_list.append(all_doc_list[i])
                        test_category_list.append(category)

    else:
        print("File doesn't exist, please run load_file_to_json first")

    return train_doc_list, train_category_list, test_doc_list, test_category_list


if __name__ == "__main__":
    # load training set and testing set from raw data.
    train_doc_list, train_category_list, test_doc_list, test_category_list = load_data_set_from_json()
    print(len(train_doc_list), len(train_category_list), len(test_doc_list), len(test_category_list))

    stopwords = load_stop_words()
    vectorizer = TfidfVectorizer(analyzer="word", tokenizer=word_segment,
                                 stop_words=stopwords)

    trainX = vectorizer.fit_transform(train_doc_list).toarray()
    trainY = train_category_list
    trainX, trainY = shuffle(trainX, trainY)

    rfc = RandomForestClassifier(n_estimators=100, criterion='gini')
    # scores = cross_val_score(svc, trainX, trainY, cv = 10, n_jobs = 10, scoring = 'accuracy')
    # print scores.mean(), scores.std()

    rfc.fit(trainX, trainY)

    testX, testY = vectorizer.transform(test_doc_list).toarray(), test_category_list
    preY = rfc.predict(testX)
    print(metrics.accuracy_score(testY, preY))
