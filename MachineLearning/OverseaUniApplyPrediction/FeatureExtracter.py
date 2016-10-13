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
        row_index_is_zero = True
        for row in reader:
            if not row_index_is_zero and row["ranking"] != "NULL":
                row_list.append(row)
            row_index_is_zero = False
        return row_list

