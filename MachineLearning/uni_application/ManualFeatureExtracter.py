import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn.utils import shuffle
from sklearn import metrics, svm, tree, naive_bayes
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class ManualFeatureExtracter:
    classifier_dict = {
        "SVM": OneVsRestClassifier(svm.SVC(kernel="linear")),
        "RFC": RandomForestClassifier(n_estimators=100, criterion='gini'),
        "DT": tree.DecisionTreeClassifier(),
        "NB": naive_bayes.GaussianNB()
    }

    def __init__(self, csv_path, test_size=0.7):
        self.file_path = csv_path
        self.test_size = test_size
        self.data = self.read_csv(self.file_path)
        self.X, self.Y = self.pre_process_data()
        self.train_set_x, self.train_set_y, self.test_set_x, self.test_set_y = self.load_data_set(ratio=test_size)

    @classmethod
    def read_csv(cls, csv_path):
        csv_file = open(csv_path, encoding="utf-8")
        reader = csv.DictReader(csv_file)

        row_list = list()
        row_index_is_zero = True
        for row in reader:
            if not row_index_is_zero and row["ranking"] != "NULL":
                row_list.append(row)
            row_index_is_zero = False
        return row_list

    def pre_process_data(self):
        data_x = list()
        data_y = list()
        for row in self.data:
            instance = {}
            instance.update(self.extract_feature_undergrad_school_level(row))
            instance.update(self.extract_feature_highest_degree(row))
            instance.update(self.extract_feature_english_level(row))
            instance.update(self.extract_feature_applied_school_ranking(row))
            instance.update(self.extract_feature_gre(row))
            instance.update(self.extract_feature_undergrad_gpa(row))
            data_x.append(instance)

            data_y.append(self.extract_result(row))
        return data_x, data_y

    def load_data_set(self, ratio):
        train_set_length = int(len(self.X) * ratio)

        train_set_x = self.X[:train_set_length]
        train_set_y = self.Y[:train_set_length]

        test_set_x = self.X[train_set_length:]
        test_set_y = self.Y[train_set_length:]
        return train_set_x, train_set_y, test_set_x, test_set_y

    def classify(self, classifier_method="RFC"):
        classifier = self.classifier_dict.get(classifier_method)

        vec = DictVectorizer()

        train_x = vec.fit_transform(self.train_set_x).toarray()
        train_y = np.array(self.train_set_y, dtype=str)

        train_x, train_y = shuffle(train_x, train_y)

        classifier.fit(train_x, train_y)

        test_x = vec.transform(self.test_set_x).toarray()
        test_y = np.array(self.test_set_y, dtype=str)
        pre_y = classifier.predict(test_x)
        print("Accuracy score is %s" % metrics.accuracy_score(test_y, pre_y))

    @staticmethod
    def extract_feature_english_level(row):
        toefl_ietls_map = {
            range(0, 32): 4,
            range(32, 35): 4.5,
            range(35, 46): 5,
            range(46, 60): 5.5,
            range(60, 79): 6,
            range(79, 94): 6.5,
            range(94, 102): 7,
            range(102, 110): 7.5,
            range(110, 115): 8,
            range(115, 118): 8.5,
            range(118, 121): 9
        }
        english_level = 0
        toefl_total = row["toefl_total"]
        if toefl_total != "NULL":
            toefl_total = float(toefl_total)

            for key, val in toefl_ietls_map.items():
                if toefl_total in key:
                    english_level = val

        ielts_total = row["ielts_total"]
        if ielts_total != "NULL":
            ielts_total = float(ielts_total)
            english_level = ielts_total if ielts_total > english_level else english_level

        return {"english_level": english_level}

    @staticmethod
    def extract_feature_undergrad_gpa(row):
        return {}

    def extract_feature_undergrad_school_level(self, row):
        school_level = str(row["undergrad_school_level"])
        if school_level in ["985", "211"]:
            school_level = school_level
        elif school_level in ["NULL", "国内其他高校", "双非"]:
            school_level = "双非"
        elif self.is_school_level_c9(school_level):
            school_level = "C9"
        else:
            school_level = "其他"
        return {"undergrad_school_level": school_level}

    @staticmethod
    def is_school_level_c9(school_level):
        if school_level.__contains__("/"):
            return True

    @staticmethod
    def extract_feature_gre(row):
        return {}

    @staticmethod
    def extract_feature_applied_school_ranking(row):
        try:
            school_ranking = int(row["ranking"])
        except ValueError:
            school_ranking = 0
        return {"applied_school_ranking": school_ranking}

    @staticmethod
    def extract_result(row):
        result = row["result"]
        if result != "被拒":
            result = "offer"
        return result

    @staticmethod
    def extract_feature_highest_degree(row):
        if row["graduated_school_level"] != "NULL":
            return {"highest_degree": "MS"}
        else:
            return {"highest_degree": "BA"}


if __name__ == "__main__":
    feature_extracter = ManualFeatureExtracter("../data/UniApplyData.csv")
    feature_extracter.classify(classifier_method="NB")
    feature_extracter.classify(classifier_method="DT")
    feature_extracter.classify(classifier_method="RFC")
