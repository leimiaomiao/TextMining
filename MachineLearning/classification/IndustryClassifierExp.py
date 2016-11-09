import csv
import glob
import json
import os
import jieba
from util.XMLParser import extract_main_info, extract_ah_info, extract_word_from_xml
from classification.Preprocessor import Preprocessor


class IndustryClassifier(object):
    WS_MS_CATEGORY_FILE_PATH = "../data/ws_ms_category.json"
    doc_word_seg_file_path = "../data/doc_word_seg.json"
    word_segment_file_root = "../data/output"

    def __init__(self):
        if not os.path.exists(self.WS_MS_CATEGORY_FILE_PATH):
            self.samples = self.load_samples_category()
        else:
            self.samples = json.load(open(self.WS_MS_CATEGORY_FILE_PATH, "r"))

        # Segment samples
        self.word_seg_samples = self.doc_word_seg()

        # Pre process samples,
        # including symbols removal, location name removal, numbers removal, human name removal, etc.
        # preprocessor = Preprocessor(self.word_seg_samples)
        # self.processed_samples = preprocessor.process()

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
    def load_ms_files(root_path="../../../msyspjs/*.xml"):
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

    def doc_word_seg(self):
        """
        load docs, and then do word segmentation.
        :return: list of tuple, (id, segment_word_list)
        """
        ws_ms_category_file_path = "../data/ws_ms_category.json"

        if os.path.exists(self.doc_word_seg_file_path):
            print("Pre segmented files found, loading...")
            doc_word_list_ = json.load(open(self.doc_word_seg_file_path, "r"))
            print("Loading finished!")
        else:
            print("Extracting basic info from files...")
            sample_list = json.load(open(ws_ms_category_file_path, "r"))

            print("Segmenting words...")
            doc_word_list_ = []
            for sample in sample_list:
                word_segment = self.seg_sample(sample)
                doc_word_list_.append((sample["ID"], word_segment))

            json.dump(doc_word_list_, open(self.doc_word_seg_file_path, "w"))
            print("Loading finished!")
        return doc_word_list_

    def seg_sample(self, sample):
        content = self.load_word_segment("%s/%s.xml" % (self.word_segment_file_root, sample["ID"]))
        word_segment = extract_word_from_xml(content)

        if len(word_segment) == 0:
            word_segment = []
            seg_list = jieba.cut(sample["BASIC_INFO"])
            for seg in seg_list:
                word_segment.append(seg)

        return word_segment

    @staticmethod
    def load_word_segment(file_path):
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                file.close()
            return content
        return ""


if __name__ == "__main__":
    classifier = IndustryClassifier()
    print(classifier.samples[0])
    print(classifier.word_seg_samples[0])

    preprocessor = Preprocessor([classifier.word_seg_samples[0]])
    processed_samples = preprocessor.process_samples()

    print(processed_samples)

    print("Dimension reduction number is %s" % (len(classifier.word_seg_samples[0][1]) - len(processed_samples[0][1])))
