import json
# from word_segment.LtpApi import word_segment
from util.StringHandler import cut_sentence
import jieba
import os


def doc_word_seg():
    """
    load docs, and then do word segmentation.
    :return:
    """
    ws_ms_category_file_path = "../data/ws_ms_category.json"
    doc_word_seg_file_path = "../data/doc_word_seg.json"

    if os.path.exists(doc_word_seg_file_path):

        print("Pre segmented files found, loading...")
        doc_word_list_ = json.load(open(doc_word_seg_file_path, "r"))
        print("Loading finished!")
    else:
        print("Extracting basic info from files...")
        sample_list = json.load(open(ws_ms_category_file_path, "r"))
        doc_list = [info["BASIC_INFO"] for info in sample_list]

        print("Segmenting words...")
        doc_word_list_ = []
        for doc in doc_list:
            word_list_ = []
            sentences = cut_sentence(doc)
            for sentence in sentences:
                seg_list = jieba.cut(sentence, cut_all=False)
                for seg in seg_list:
                    if seg not in word_list_:
                        word_list_.append(seg)

            doc_word_list_.append(word_list_)

        json.dump(doc_word_list_, open(doc_word_seg_file_path, "w"))
        print("Loading finished!")
    return doc_word_list_


def symbols_removal(doc_word_list_):
    """
    remove symbols and numeric strings in doc_word_list
    :param doc_word_list_:
    :return:
    """
    doc_word_seg_symbols_removal_file_path = "../data/doc_word_seg_symbols_removal.json"
    print("Removing symbols and numeric strings from files...")
    if os.path.exists(doc_word_seg_symbols_removal_file_path):
        new_doc_word_list_ = json.load(open(doc_word_seg_symbols_removal_file_path))
    else:
        stopwords_symbols_file_path = "../stopwords_symbols.txt"
        symbols = [line.decode("utf-8").strip() for line in open(stopwords_symbols_file_path, "rb").readlines()]

        new_doc_word_list_ = []
        for word_list_ in doc_word_list_:
            new_word_list_ = []
            for word_ in word_list_:
                if word_ not in symbols and not str(word_).isnumeric():
                    new_word_list_.append(word_)
            new_doc_word_list_.append(new_word_list_)

        json.dump(new_doc_word_list_, open(doc_word_seg_symbols_removal_file_path, "w"))
    print("Symbols removal finished!")
    return new_doc_word_list_


def calc_df(doc_word_list_):
    print("Calculating document frequency...")

    df_file_path = "../data/document_frequency.json"
    if os.path.exists(df_file_path):
        word_frequency_sorted_ = json.load(open(df_file_path))
    else:
        # TODO trunk doc_word_list
        # TODO invoke multi-thread to handle each trunk
        # TODO Join result
        word_frequency_ = calc_trunk_df(doc_word_list_, 1)

        word_frequency_sorted_ = sort_frequency(word_frequency_)
        json.dump(word_frequency_sorted_, open(df_file_path, "w"))
    print("Calculation finished!")
    return word_frequency_sorted_


def sort_frequency(word_frequency_):
    print("Sorting frequency...")
    frequency_sorted_ = sorted(word_frequency_, key=lambda key: key[1], reverse=True)
    print("Sorting frequency finished!")
    return frequency_sorted_


def calc_trunk_df(doc_word_list_trunk, trunk_index):
    print("Calculating document frequency of trunk %s" % trunk_index)

    trunk_temp_file_path = "../data/trunk/%s.json" % trunk_index

    word_frequency_ = {}
    for word_list_ in doc_word_list_trunk:

        for word in get_word_set(word_list_):
            if word in word_frequency_.keys():
                word_frequency_[word] += 1
            else:
                word_frequency_[word] = 1

    word_frequency_list = list(word_frequency_.items())
    json.dump(word_frequency_list, open(trunk_temp_file_path, "w"))
    print("Calculating document frequency of trunk %s finished" % trunk_index)
    return word_frequency_list


def get_word_set(word_list):
    word_set = []
    for word in word_list:
        word = str(word).lstrip()
        if word not in word_set and len(word) > 0:
            word_set.append(word)
    return word_set


if __name__ == "__main__":
    doc_word_list = doc_word_seg()
    doc_word_list = symbols_removal(doc_word_list)
    word_frequency = calc_df(doc_word_list)

