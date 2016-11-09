import csv
import json
import os
from classification.IndustryClassifierExp import IndustryClassifier


class Preprocessor:
    def __init__(self):
        self.symbol_list = self.get_symbol_list()
        self.location_name_list = self.get_location_name_list()

    @staticmethod
    def get_symbol_list():
        stopwords_symbols_file_path = "../stopwords/stopwords_symbols.json"

        if not os.path.exists(stopwords_symbols_file_path):
            symbols_file_path = "../stopwords_symbols.txt"
            symbols = [line.decode("utf-8").strip() for line in open(symbols_file_path, "rb").readlines()]
            json.dump(symbols, open(stopwords_symbols_file_path, "w"))

        else:
            symbols = json.load(open(stopwords_symbols_file_path, "r"))

        return symbols

    @staticmethod
    def get_location_name_list():
        stopwords_location_names_file_path = "../stopwords/stopwords_location_names.json"

        if not os.path.exists(stopwords_location_names_file_path):
            location_names_file_path = "../location_names.csv"

            name_list = []
            with open(location_names_file_path, "r", encoding="utf-8") as file:
                reader = csv.reader(file)
                for row in reader:
                    province = str(row[0]).lstrip()
                    if len(province) > 0 and province not in name_list:
                        name_list.append(province)
                    if "省" in province:
                        province_abbr = province.replace("省", "")
                        if province_abbr not in name_list:
                            name_list.append(province_abbr)

                    city = str(row[1]).lstrip()
                    if len(city) > 0 and city not in name_list:
                        name_list.append(city)
                    if "市" in city:
                        city_abbr = city.replace("市", "")
                        if city_abbr not in name_list:
                            name_list.append(city_abbr)

                    town = str(row[2]).lstrip()
                    if len(town) > 0 and town not in name_list:
                        name_list.append(town)

            json.dump(name_list, open(stopwords_location_names_file_path, "w"))
        else:
            name_list = json.load(open(stopwords_location_names_file_path, "r"))
        return name_list

    @staticmethod
    def word_segment(sentence):
        pass

    def remove_symbols(self, word_list):
        new_word_list = []

        for word in word_list:
            if word not in self.symbol_list:
                new_word_list.append(word)

        return new_word_list

    def remove_location_names(self, word_list):
        new_word_list = []

        for word in word_list:
            if word not in self.location_name_list:
                new_word_list.append(word)

        return new_word_list

    def remove_stopwords(self, word_list):
        pass

if __name__ == "__main__":
    industry_classifier = IndustryClassifier()
    # for sample in industry_classifier.samples:
    #     # with open("/Users/leimiaomiao/Documents/Projects/Postgraduate/TextMining/input/%s.txt" % sample["ID"], "wb") as file:
    #     #     file.write(str(sample["BASIC_INFO"]).encode("utf-8"))
    #     # file.close()
    #
    #     os.system("cat %s/%s.txt | %s --segmentor-model %s --last-stage ws > %s/%s.txt" % (
    #         "../../input",
    #         sample["ID"],
    #         "../../ltp/bin/ltp_test",
    #         "../../ltp/ltp_data/cws.model",
    #         "../../output",
    #         sample["ID"])
    #     )
    #     break


