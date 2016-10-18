import csv
import jieba
import jieba.posseg as pseg
import pycrfsuite


class ProductExtract(object):
    def __init__(self, sample_path="../data/500TaggedSamples.csv"):
        self.sample_path = sample_path
        self.instance_list, self.tagged_product_list = self.load_samples(sample_path)
        self.user_dict = self.load_user_dict()
        jieba.load_userdict(self.user_dict)
        self.tagged_instance_list = self.pos_tag()

    @staticmethod
    def load_samples(sample_path):
        instance_list = list()
        tagged_product_list = list()

        with open(sample_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            index = 0
            for row in reader:
                if index != 0:
                    instance_list.append(row[1])
                    tagged_product_list.append(row[2])
                index += 1
        return instance_list, tagged_product_list

    def load_user_dict(self):
        user_dict = list()
        for product in self.tagged_product_list:
            for p in str(product).split("、"):
                if len(p.lstrip()) > 0:
                    user_dict.append(p.lstrip())
        return user_dict

    @staticmethod
    def cut_sentence(words):
        start = 0
        i = 0
        sentence_list = []
        punt_list = str(',.!?:;~，。！？：；～')
        for word in words:
            if word in punt_list and token not in punt_list:  # 检查标点符号下一个字符是否还是标点
                sentence_list.append(words[start:i + 1])
                start = i + 1
                i += 1
            else:
                i += 1
                token = list(words[start:i + 2]).pop()  # 取下一个字符
        if start < len(words):
            sentence_list.append(words[start:])
        return sentence_list

    @staticmethod
    def word2features(sent, i):
        pass

    @staticmethod
    def sent2features(sent):
        pass

    @staticmethod
    def sent2labels(sent):
        pass

    @staticmethod
    def sent2tokens(sent):
        pass

    def pos_tag(self):
        instance_i = 0
        tagged_instance = []
        for instance in self.instance_list:
            sentence_list = self.cut_sentence(instance)

            sent_i = 0  # 记录词语出现在第几个句子
            tagged_each_instance = []
            for sent in sentence_list:
                words = pseg.cut(sent)

                for token, pos in words:
                    if token == self.tagged_product_list[instance_i]:
                        tagged_each_instance.append((token, pos, sent_i, True))
                    else:
                        tagged_each_instance.append((token, pos, sent_i, False))
                sent_i += 1

            instance_i += 1
            tagged_instance.append(tagged_each_instance)
        return tagged_instance


if __name__ == "__main__":
    product_extract = ProductExtract()

    # print(" ".join(jieba.cut(product_extract.instance_list[0])))
    # print(" ".join(jieba.cut(product_extract.tagged_product_list[0])))
    #
    # for s in product_extract.cut_sentence(product_extract.instance_list[0]):
    #     print(s + "\n")
    #
    # words = pseg.cut(product_extract.instance_list[0])
    # for word, flag in words:
    #     print(word, flag)
