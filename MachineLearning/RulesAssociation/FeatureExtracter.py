import csv


class FeatureExtracter:
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
        english_level_list = list()
        for row in self.data:
            english_level = 0
            toefl_total = row["toefl_total"]
            if toefl_total != "NULL":
                toefl_total = int(toefl_total)
                if 118 <= toefl_total <= 120:
                    english_level = 9.0
                elif 115 <= toefl_total <= 117:
                    english_level = 8.5
                elif 110 <= toefl_total <= 114:
                    english_level = 8.0
                elif 102 <= toefl_total <= 109:
                    english_level = 7.5
                elif 94 <= toefl_total <= 101:
                    english_level = 7.0
                elif 79 <= toefl_total <= 93:
                    english_level = 6.5
                elif 60 <= toefl_total <= 78:
                    english_level = 6.0
                elif 46 <= toefl_total <= 59:
                    english_level = 5.5
                elif 35 <= toefl_total <= 45:
                    english_level = 5.0
                elif 32 <= toefl_total <= 34:
                    english_level = 4.5
                else:
                    english_level = 4

            ielts_total = row["ielts_total"]
            if ielts_total != "NULL":
                ielts_total = float(ielts_total)
                english_level = ielts_total if ielts_total > english_level else english_level

            english_level_list.append(english_level)

        return english_level_list


if __name__ == "__main__":
    feature_extracter = FeatureExtracter("query.csv")
    english_level_list = feature_extracter.extract_feature_english_level()
    print(len(english_level_list))
