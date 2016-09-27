from util.XMLParser import extract_main_info


def read_main_info(from_file_path, extract, decode="utf-8"):
    with open(from_file_path, "rb") as f:
        doc = f.read().decode(decode)
        if extract:
            doc = extract_main_info(doc)
        return doc


def load_stop_words():
    path = "../stopwords.txt"
    stop_words = [line.decode("gbk").strip() for line in open(path, "rb").readlines()]
    stop_words.append("原告")
    stop_words.append("被告")
    return stop_words
