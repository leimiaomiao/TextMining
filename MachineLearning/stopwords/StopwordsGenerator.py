import json
import os
from util.XMLParser import extract_word_from_xml
import jieba


def doc_word_seg():
    """
    load docs, and then do word segmentation.
    :return:
    """
    ws_ms_category_file_path = "../data/ws_ms_category.json"
    doc_word_seg_file_path = "../data/doc_word_seg.json"
    word_segment_file_root = "../data/output"

    if os.path.exists(doc_word_seg_file_path):

        print("Pre segmented files found, loading...")
        doc_word_list_ = json.load(open(doc_word_seg_file_path, "r"))
        print("Loading finished!")
    else:
        print("Extracting basic info from files...")
        sample_list = json.load(open(ws_ms_category_file_path, "r"))

        print("Segmenting words...")
        doc_word_list_ = []
        for sample in sample_list:
            content = load_word_segment("%s/%s.xml" % (word_segment_file_root, sample["ID"]))
            word_segment = extract_word_from_xml(content)

            if len(word_segment) == 0:
                word_segment = jieba.cut(sample["BASIC_INFO"])

            doc_word_list_.append((sample["ID"], word_segment))

        json.dump(doc_word_list_, open(doc_word_seg_file_path, "w"))
        print("Loading finished!")
    return doc_word_list_


def load_word_segment(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            file.close()
        return content
    return ""


def duplicate_words_removal(doc_word_list_):
    print("Starting duplicate words removal...")
    unique_doc_word_seg_file_path = "../data/unique_doc_word_seg.json"
    if os.path.exists(unique_doc_word_seg_file_path):
        unique_doc_word_list_ = json.load(open(unique_doc_word_seg_file_path, "r"))
    else:
        unique_doc_word_list_ = []
        for _id, word_list_ in doc_word_list_:
            unique_word_list = get_word_set(word_list_)
            unique_doc_word_list_.append((_id, unique_word_list))
        json.dump(unique_doc_word_list_, open(unique_doc_word_seg_file_path, "w"))

    print("Duplicate words removal finished!")
    return unique_doc_word_list_


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
        for _id, word_list_ in doc_word_list_:
            new_word_list_ = []
            for word_ in word_list_:
                if word_ not in symbols and not str(word_).isnumeric():
                    new_word_list_.append(word_)
            new_doc_word_list_.append((_id, new_word_list_))

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
    for _, word_list_ in doc_word_list_trunk:
        for word in word_list_:
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
    doc_word_list = duplicate_words_removal(doc_word_list)
    word_frequency = calc_df(doc_word_list)

    with open("../words_frequency.txt", "w") as file:
        for wf in word_frequency:
            file.write("%s,%s" % (wf[0], wf[1]))
            file.write("\n")
        file.close()
