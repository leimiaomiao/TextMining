import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from util.FileReader import read_main_info, load_stop_words
from word_segment.WordSegment import word_segment


TRUE_K = 8


def load_data(root_path, extract=True, decode="utf-8"):
    docs = list()
    file_list = glob.glob(root_path)

    for file in file_list:
        docs.append(read_main_info(file, extract=extract, decode=decode))

    return docs


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
