import csv
import glob
import json
import os
import jieba
from util.XMLParser import extract_main_info, extract_ah_info, extract_word_from_xml
from classification.Preprocessor import Preprocessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from sklearn import metrics, svm, tree, naive_bayes
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA, TruncatedSVD
import numpy as np
from datetime import datetime


class IndustryClassifier(object):
    WS_MS_CATEGORY_FILE_PATH = "../data/ws_ms_category.json"
    doc_word_seg_file_path = "../data/doc_word_seg.json"
    word_segment_file_root = "../data/output"
    preprocessor_file_path = "../data/preprocessed.json"
    industry_classification_dataset_file_path = "../data/industry_classification_dataset.json"
    industry_classification_dataset_without_preprocess_file_path = "../data/industry_classification_dataset_unpreprocess.json"
    category_sample_map_file_path = "../data/category_sample_map.json"
    category_sample_without_preprocess_map_file_path = "../data/category_sample_map_unpreprocess.json"

    classifier_dict = {
        "SVM": OneVsRestClassifier(svm.SVC(kernel="linear")),
        "RFC": RandomForestClassifier(n_estimators=100, criterion='gini'),
        "DT": tree.DecisionTreeClassifier(),
        "NB": naive_bayes.GaussianNB()
    }

    def __init__(self, preprocess=True, min_df=0):
        self.preprocess = preprocess
        if not os.path.exists(self.WS_MS_CATEGORY_FILE_PATH):
            self.samples = self.load_samples_category()
        else:
            self.samples = json.load(open(self.WS_MS_CATEGORY_FILE_PATH, "r"))

        # Segment samples
        self.word_seg_samples = self.doc_word_seg()

        # Pre process samples,
        # including symbols removal, location name removal, numbers removal, human name removal, etc.
        if self.preprocess:
            if not os.path.exists(self.preprocessor_file_path):
                preprocessor = Preprocessor()
                self.processed_samples = preprocessor.process_samples(self.word_seg_samples)
                json.dump(self.processed_samples, open(self.preprocessor_file_path, "w"))
            else:
                self.processed_samples = json.load(open(self.preprocessor_file_path, "r"))
        else:
            self.processed_samples = self.word_seg_samples

        self.sample_category_map = self.get_sample_category_map(preprocess=self.preprocess)
        self.train_set, self.train_set_category, self.test_set, self.test_set_category = self.load_data_set()
        self.min_df = min_df

    def load_data_set(self, ratio=0.7):
        print("Start loading dataset...")
        if self.preprocess:
            file = self.industry_classification_dataset_file_path
        else:
            file = self.industry_classification_dataset_without_preprocess_file_path
        if os.path.exists(file):
            data = json.load(open(file, "r"))
            train_set = data["train_set"]
            train_set_category = data["train_set_category"]
            test_set = data["test_set"]
            test_set_category = data["test_set_category"]
        else:
            train_set = []
            train_set_category = []
            test_set = []
            test_set_category = []

            category_list = self.sample_category_map.keys()
            for category in category_list:
                sample_list = self.sample_category_map.get(category)
                train_set_length = int(len(sample_list) * ratio)

                index = 0
                for sample in sample_list:
                    if index < train_set_length:
                        train_set.append(sample)
                        train_set_category.append(category)
                    else:
                        test_set.append(sample)
                        test_set_category.append(category)
                    index += 1

            data = {
                "train_set": train_set,
                "train_set_category": train_set_category,
                "test_set": test_set,
                "test_set_category": test_set_category
            }
            json.dump(data, open(file, "w"))
        print("Loading dataset finished!")
        return train_set, train_set_category, test_set, test_set_category

    def get_sample_category_map(self, preprocess):
        print("Getting sample category map...")
        if preprocess:
            file = self.category_sample_map_file_path
        else:
            file = self.category_sample_without_preprocess_map_file_path

        if os.path.exists(file):
            category_map = json.load(open(file, "r"))
        else:
            category_map = {}
            for sample in self.processed_samples:
                _id = sample[0]
                word_seg_sample = " ".join(sample[1])
                category = str(self.get_category_by_id(_id)).lstrip()
                if len(category) < 2:
                    category = "其他行业"
                if category in category_map.keys():
                    sample_list = category_map.get(category)
                    sample_list.append(word_seg_sample)
                else:
                    category_map[category] = list()
                    category_map[category].append(word_seg_sample)
            json.dump(category_map, open(file, "w"))

        return category_map

    def get_category_by_id(self, _id):
        for sample in self.samples:
            if sample["ID"] == _id:
                return sample["CATEGORY"]
        return "其他行业"

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

    def classify(self, classifier_method="RFC"):
        start_time = datetime.now()
        classifier = self.classifier_dict.get(classifier_method)

        vec = TfidfVectorizer(self.min_df)
        # svd = TruncatedSVD(n_components=1000, n_iter=7, random_state=42)
        svd = PCA(n_components=1500, svd_solver='full')
        train_x = vec.fit_transform(self.train_set).toarray()
        train_x = svd.fit_transform(train_x)
        train_y = np.array(self.train_set_category, dtype=str)
        # train_x, train_y = shuffle(train_x, train_y)

        classifier.fit(train_x, train_y)
        test_x = vec.transform(self.test_set).toarray()
        test_y = np.array(self.test_set_category, dtype=str)
        test_x = svd.transform(test_x)
        pre_y = classifier.predict(test_x)
        report = classification_report(test_y, pre_y)
        accuracy = accuracy_score(test_y, pre_y)
        end_time = datetime.now()
        print("Time expense: %s" % (end_time - start_time))
        print("Accuracy: %s" % accuracy)
        print(report)
        return report


if __name__ == "__main__":
    industry_classifier = IndustryClassifier(preprocess=True, min_df=10)

    # map = industry_classifier.sample_category_map
    # for key in map.keys():
    #     print("%s : %s" % (key, len(map.get(key))))

    print("Test 1: with text preprocessing: %s" % industry_classifier.preprocess)
    print("Min_df = %s" % industry_classifier.min_df)
    industry_classifier.classify(classifier_method="NB")
    industry_classifier.classify(classifier_method="DT")
    industry_classifier.classify(classifier_method="RFC")
    industry_classifier.classify(classifier_method="SVM")
