import json
import csv


class ManualFeatureExtracter:
    def __init__(self, csv_path):
        self.file_path = csv_path
        self.data = self.read_csv(self.file_path)

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
        instance_list = list()
        for row in self.data:
            instance = {}
            instance.update(self.extract_feature_undergrad_school_level(row))
            instance.update(self.extract_feature_highest_degree(row))
            instance.update(self.extract_feature_english_level(row))
            instance.update(self.extract_feature_applied_school_ranking(row))
            instance.update(self.extract_feature_gre(row))
            instance.update(self.extract_feature_undergrad_gpa(row))
            instance.update(self.extract_result(row))
            instance_list.append(instance)
        return instance_list

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
        return {"result": row["result"]}

    @staticmethod
    def extract_feature_highest_degree(row):
        if row["graduated_school_level"] != "NULL":
            return {"highest_degree": "MS"}
        else:
            return {"highest_degree": "BA"}


if __name__ == "__main__":
    feature_extracter = ManualFeatureExtracter("../data/UniApplyData.csv")
    instances = feature_extracter.pre_process_data()
    with open("../data/UniApplyInstances.json", "w") as file:
        json.dump(instances, file)

