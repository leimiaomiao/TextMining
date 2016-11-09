import csv
import json
import os
import re


class Preprocessor:
    def __init__(self, word_seg_samples=[]):
        self.symbol_list = self.get_symbol_list()
        self.location_name_list = self.get_location_name_list()
        self.word_seg_samples = word_seg_samples

        self.procedures = [
            self.remove_symbols,
            self.remove_numbers,
            self.remove_location_names,
            self.remove_human_names,
            self.remove_stopwords
        ]

    def process_samples(self):
        processed_samples = []
        for sample in self.word_seg_samples:
            _id = sample[0]
            word_seg_samples = sample[1]

            processed_word_seg_sample = []
            for word in word_seg_samples:
                for procedure in self.procedures:
                    word = procedure(word)
                if len(word) > 0:
                    processed_word_seg_sample.append(word)

            processed_samples.append((_id, processed_word_seg_sample))

        return processed_samples

    def process_word(self, word):
        for procedure in self.procedures:
            word = procedure(word)
        return word

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

    def remove_symbols(self, word):
        if len(word) > 0 and word not in self.symbol_list:
            return word
        return ""

    def remove_location_names(self, word):
        if len(word) > 0 and word not in self.location_name_list \
                and not str(word).endswith("县") \
                and not str(word).endswith("区") \
                and not str(word).endswith("村") \
                and not str(word).endswith("镇"):
            return word
        return ""

    def remove_stopwords(self, word):
        return word

    def remove_human_names(self, word):
        return word

    @staticmethod
    def remove_numbers(word):
        if len(word) > 0 and not str(word).isnumeric() and not str(word).isdecimal():
            pattern = re.compile(r'.*\d+')
            result = pattern.match(word)
            if result is None:
                return word
        return ""
