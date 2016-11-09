import csv
import glob
import json
import os
from util.XMLParser import extract_main_info, extract_ah_info


class IndustryClassifier(object):
    WS_MS_CATEGORY_FILE_PATH = "../data/ws_ms_category.json"

    def __init__(self):
        if not os.path.exists(self.WS_MS_CATEGORY_FILE_PATH):
            self.samples = self.load_samples_category()
        else:
            self.samples = json.load(open(self.WS_MS_CATEGORY_FILE_PATH, "r"))

    def load_samples_category(self, file_path="../data/wstool_ms.csv"):
        ms_info_dict = self.load_ms_files()

        sample_list = []
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            index = 0
            for row in reader:

                case_id = str(row[1]).lstrip()
                category = str(row[2]).lstrip()
                if case_id in ms_info_dict.keys():
                    ms_basic_info = ms_info_dict[case_id]
                    index += 1
                    sample = {
                        "ID": index,
                        "CASE_ID": case_id,
                        "BASIC_INFO": ms_basic_info,
                        "CATEGORY": category
                    }
                    sample_list.append(sample)

        with open(self.WS_MS_CATEGORY_FILE_PATH, "w") as file:
            json.dump(sample_list, file)

        return sample_list

    @staticmethod
    def load_ms_files(root_path="/Users/leimiaomiao/Documents/Projects/Postgraduate/msyspjs/*.xml"):
        docs = {}
        file_list = glob.glob(root_path)

        for file in file_list:
            with open(file, "rb") as f:
                try:
                    doc = f.read().decode("utf-8")
                except UnicodeDecodeError:
                    doc = f.read().decode("latin-1")

                main_info = extract_main_info(doc).lstrip()
                ah_info = extract_ah_info(doc).lstrip()

                if main_info != "" and ah_info != "":
                    docs.update({ah_info: main_info})

        return docs


if __name__ == "__main__":
    classifier = IndustryClassifier()
    print(len(classifier.samples))
    # doc_word_seg = json.load(open("../data/doc_word_seg.json"))
    # print(doc_word_seg[0])
