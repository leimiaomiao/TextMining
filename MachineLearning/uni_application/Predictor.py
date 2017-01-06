import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn.utils import shuffle
from sklearn import metrics, svm, tree, naive_bayes
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class Predictor:
    classifier_dict = {
        "SVM": OneVsRestClassifier(svm.SVC(kernel="linear")),
        "RFC": RandomForestClassifier(n_estimators=100, criterion='gini'),
        "DT": tree.DecisionTreeClassifier(),
        "NB": naive_bayes.GaussianNB()
    }

    school_level_map = {
        "C9": 1,
        "985": 2,
        "211": 3,
        "双非": 4,
        "其他": 5
    }

    degree_map = {
        "BA": 0,
        "MS": 1
    }

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

    def __init__(self, csv_path, test_size=0.7):
        self.file_path = csv_path
        self.test_size = test_size
        self.data = self.read_csv(self.file_path)
        self.X, self.Y = self.extract_features()
        # self.X, self.Y = self.pre_process_data()
        self.train_set_x, self.train_set_y, self.test_set_x, self.test_set_y = self.load_data_set(ratio=test_size)

    @classmethod
    def read_csv(cls, csv_path):
        csv_file = open(csv_path, encoding="utf-8")
        reader = csv.DictReader(csv_file)

        row_list = list()
        row_index_is_zero = True
        for row in reader:
            if not row_index_is_zero:
                # Hard code the school ranking of "The University of Hong Kong"
                if str(row["school_name"]) == "The University of Hong Kong" and \
                                row["apply_school_qs_ranking"] == "NULL":
                    row["apply_school_qs_ranking"] = 30
                # Remove instances with no school ranking
                if row["apply_school_qs_ranking"] != "NULL":
                    row_list.append(row)
            row_index_is_zero = False
        return row_list

    def extract_features(self):
        data_x = list()
        data_y = list()
        for row in self.data:
            instance = {}

            if self.extract_result(row) is True:
                instance.update({"undergrad_school_level": self.extract_feature_undergrad_school_level(row)})
                instance.update({"highest_degree": self.extract_feature_highest_degree(row)})
                instance.update({"english_level": self.extract_feature_english_level(row)})
                instance.update({"gre": self.extract_feature_gre(row)})
                instance.update({"undergrad_gpa": self.extract_feature_undergrad_gpa(row)})
                data_x.append(instance)

                data_y.append({"applied_school_ranking": self.extract_feature_applied_school_ranking(row)})
        return data_x, data_y

    def normalize_data(self):
        school_level_min = min(self.school_level_map.values())
        school_level_max = max(self.school_level_map.values())

        english_level_min = min(self.toefl_ietls_map.values())
        english_level_max = max(self.toefl_ietls_map.values())

        gpa_score_list = [x["undergrad_gpa"] for x in self.X]
        gpa_score_min = min(gpa_score_list)
        gpa_score_max = max(gpa_score_list)

        gre_list = [x["gre"] for x in self.X]
        gre_min = min(gre_list)
        gre_max = max(gre_list)

        applied_school_ranking_list = [y["applied_school_ranking"] for y in self.Y]
        applied_school_ranking_min = min(applied_school_ranking_list)
        applied_school_ranking_max = max(applied_school_ranking_list)

        for x in self.X:
            # normalize undergrad_school_level
            school_level = x["undergrad_school_level"]
            normalized_school_level = (school_level - school_level_min) / (school_level_max - school_level_min)
            x["undergrad_school_level"] = normalized_school_level

            # normalize highest_degree
            x["highest_degree"] = self.degree_map[x["highest_degree"]]

            # normalize english_level
            english_level = x["english_level"]
            normalized_english_level = (english_level - english_level_min) / (english_level_max - english_level_min)
            x["english_level"] = normalized_english_level

            # normalize gre_score
            gre = x["gre"]
            normalized_gre = (gre - gre_min) / (gre_max - gre_min)
            x["gre"] = normalized_gre

            # normalize gpa_score
            gpa_score = x["undergrad_gpa"]
            normalized_gpa_score = (gpa_score - gpa_score_min) / (gpa_score_max - gpa_score_min)
            x["undergrad_gpa"] = normalized_gpa_score

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

    def extract_feature_english_level(self, row):
        english_level = 0
        toefl_total = row["toefl_total"]
        if toefl_total != "NULL":
            toefl_total = float(toefl_total)

            for key, val in self.toefl_ietls_map.items():
                if toefl_total in key:
                    english_level = val

        ielts_total = row["ielts_total"]
        if ielts_total != "NULL":
            ielts_total = float(ielts_total)
            english_level = ielts_total if ielts_total > english_level else english_level

        return english_level

    def extract_feature_undergrad_gpa(self, row):
        undergrad_average_score = row["undergrad_average_score"]
        undergrad_gpa = row["undergrad_gpa"]
        undergrad_gpa_base = row["undergrad_gpa_base"]

        if undergrad_average_score != "NULL":
            try:
                undergrad_average_score = float(undergrad_average_score)
                score = float(undergrad_average_score) / 100
            except ValueError:
                score = self.calc_score_by(undergrad_gpa, undergrad_gpa_base)
        else:
            score = self.calc_score_by(undergrad_gpa, undergrad_gpa_base)

        return round(score, 2)

    @staticmethod
    def calc_score_by(undergrad_gpa, undergrad_gpa_base):
        gpa = 0
        if undergrad_gpa != "NULL":
            try:
                gpa = float(undergrad_gpa)
            except ValueError:
                gpa = 0

        if undergrad_gpa_base != "NULL":
            try:
                base = float(undergrad_gpa_base)
            except ValueError:
                if gpa < 4:
                    base = 4
                else:
                    base = 5
        else:
            if gpa < 4:
                base = 4
            else:
                base = 5
        score = gpa / base
        return score

    def extract_feature_undergrad_school_level(self, row):
        school_level = str(row["undergrad_school_level"])
        if school_level in ["985", "211"]:
            school_level = school_level
        elif school_level in ["国内其他高校", "双非"]:
            school_level = "双非"
        elif self.is_school_level_c9(school_level):
            school_level = "C9"
        else:
            school_level = "其他"
        return self.school_level_map[school_level]

    @staticmethod
    def is_school_level_c9(school_level):
        if school_level.__contains__("/"):
            return True

    @staticmethod
    def extract_feature_gre(row):
        gre_total = row["gre_total"]
        if gre_total != "NULL":
            try:
                gre_total = float(gre_total)
            except ValueError:
                gre_total = 0
        else:
            gre_total = 0
        return gre_total

    @staticmethod
    def extract_feature_applied_school_ranking(row):
        try:
            school_ranking = int(row["apply_school_qs_ranking"])
        except ValueError:
            school_ranking = 0
        return school_ranking

    @staticmethod
    def extract_result(row):
        result = row["result"]
        if result == "被拒":
            return False
        return True

    @staticmethod
    def extract_feature_highest_degree(row):
        if row["graduated_school_level"] != "NULL":
            return "MS"
        return "BA"


if __name__ == "__main__":
    predictor = Predictor("uni_apply_data.csv")
    # predictor.classify(classifier_method="RFC")
    # predictor.normalize_data()

    applied_school_ranking_list = [y["applied_school_ranking"] for y in predictor.Y]
    applied_school_ranking_min = min(applied_school_ranking_list)
    applied_school_ranking_max = max(applied_school_ranking_list)

    print(sorted(applied_school_ranking_list, reverse=True))
    # print(applied_school_ranking_max)
