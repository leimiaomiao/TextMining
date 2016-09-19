import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import xml.etree.ElementTree as ET

TRUE_K = 8


def load_data(root_path):
    docs = list()
    file_list = glob.glob(root_path)

    for file in file_list:
        with open(file, 'rb') as f:
            doc = f.read().decode("utf-8")
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


def word_segment(doc):
    pass


if __name__ == "__main__":
    print("Loading data...")
    corpus = load_data("../公开民事文书xml/*.xml")

    print("Extracting feature...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    print("Executing clustering...")
    km = KMeans(n_clusters=TRUE_K)
    km.fit(X)
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(TRUE_K):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()
