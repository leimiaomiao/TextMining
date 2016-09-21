import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
import xml.etree.ElementTree as ET
from word_segment.WordSegment import word_segment

TRUE_K = 8


def load_stop_words():
    path = "../stopwords.txt"
    stop_words = [line.decode("gbk").strip() for line in open(path, "rb").readlines()]
    stop_words.append("原告")
    stop_words.append("被告")
    return stop_words


def load_data(root_path, extract=True, decode="utf-8"):
    docs = list()
    file_list = glob.glob(root_path)

    for file in file_list:
        with open(file, 'rb') as f:
            doc = f.read().decode(decode)
            if extract:
                doc = extract_main_info(doc)
            docs.append(doc)
        f.close()

    return docs


def extract_main_info(doc):
    root = ET.fromstring(doc)
    info = root.find(".//AJJBQK[@nameCN='案件基本情况']")
    if info is not None:
        main_info = info.get("value")
        return main_info
    else:
        return ""


if __name__ == "__main__":
    print("Loading data...")
    corpus = load_data("../公开民事文书xml/*.xml")
    stopwords = load_stop_words()

    print("Extracting feature...")
    vectorizer = TfidfVectorizer(analyzer="word", tokenizer=word_segment,
                                 stop_words=stopwords)
    X = vectorizer.fit_transform(corpus)

    print("Executing clustering...")
    km = MiniBatchKMeans(n_clusters=TRUE_K)
    km.fit(X)
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(TRUE_K):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()
