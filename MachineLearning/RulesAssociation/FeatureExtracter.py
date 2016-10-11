import csv


class FeatureExtracter:
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

    def __init__(self, csv_path):
        self.file_path = csv_path
        self.data = self.read_csv(self.file_path)

    @classmethod
    def read_csv(cls, csv_path):
        csv_file = open(csv_path, encoding="utf-8")
        reader = csv.DictReader(csv_file)

        row_list = list()
        row_index = 0
        for row in reader:
            if row_index != 0:
                row_list.append(row)
            row_index += 1

        return row_list

    def extract_feature_english_level(self):
        _english_level_list = list()
        for row in self.data:
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

            _english_level_list.append(english_level)

        return _english_level_list

    def extract_feature_undergrad_gpa(self):
        pass


if __name__ == "__main__":
    feature_extracter = FeatureExtracter("query.csv")
    english_level_list = feature_extracter.extract_feature_english_level()
    print(len(english_level_list))
    print(feature_extracter.data[0])
