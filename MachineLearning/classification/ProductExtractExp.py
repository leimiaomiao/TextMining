import csv
import jieba
import jieba.posseg as pseg
import pycrfsuite
import os
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from itertools import chain


class ProductExtract(object):
    def __init__(self, sample_path="../data/500TaggedSamples.csv"):
        self.sample_path = sample_path
        self.instance_list, self.tagged_product_list = self.load_samples(sample_path)
        self.user_dict = self.load_user_dict()
        jieba.load_userdict(self.user_dict)

        self.tagged_instance_list = self.pos_tag()

        self.ratio = 0.7
        self.trainset_length = int(len(self.tagged_instance_list) * self.ratio)

        self.x_train = [self.instance2features(instance) for instance in
                        self.tagged_instance_list[:self.trainset_length]]
        self.y_train = [self.instance2labels(instance) for instance in self.tagged_instance_list[:self.trainset_length]]

        self.x_test = [self.instance2features(instance) for instance in
                       self.tagged_instance_list[self.trainset_length:]]
        self.y_test = [self.instance2labels(instance) for instance in self.tagged_instance_list[self.trainset_length:]]

        self.train_model_path = '../data/product.crfsuite'

        if not os.path.exists(self.train_model_path):
            self.train_model()

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
                    user_dict.append("%s %s" % (p.lstrip(), "n"))
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
    def word2features(instance, i):
        pos_tag = instance[i]
        features = [
            "word.token=%s" % pos_tag[0],
            "word.pos_tag=%s" % pos_tag[1],
            "word.sent_index=%s" % pos_tag[2],
            "word.reverse_word_index=%s" % pos_tag[3],
            "word.length=%s" % pos_tag[4]
        ]
        # print(features)
        return features

    def instance2features(self, instance):
        return [self.word2features(instance, i) for i in range(len(instance))]

    @staticmethod
    def instance2labels(instance):
        return [tag[len(tag) - 1] for tag in instance]

    @staticmethod
    def instance2tokens(instance):
        return [tag[0] for tag in instance]

    def pos_tag(self):
        instance_i = 0
        tagged_instance = []
        for instance in self.instance_list:
            sentence_list = self.cut_sentence(instance)

            sent_i = 0  # 记录词语出现在第几个句子
            tagged_each_instance = []
            for sent in sentence_list:
                words = pseg.cut(sent)
                tokenize_list = jieba.tokenize(sent)
                words_len = len(list(tokenize_list))

                word_i = 0
                for token, pos in words:
                    reverse_word_i = words_len - word_i  # 记录词语是句子中的倒数第几个词语
                    token_len = len(token)  # 记录词语的长度

                    if token == self.tagged_product_list[instance_i]:
                        tagged_each_instance.append((token, pos, sent_i, reverse_word_i, token_len, "PRODUCT"))
                    else:
                        tagged_each_instance.append((token, pos, sent_i, reverse_word_i, token_len, "NONE-PRODUCT"))
                    word_i += 1
                sent_i += 1

            instance_i += 1
            tagged_instance.append(tagged_each_instance)

        # for token, pos, sent_i, reverse_word_i, token_len, is_product in tagged_instance[0]:
        #     print((token, pos, sent_i, reverse_word_i, token_len, is_product))
        return tagged_instance

    def train_model(self):
        trainer = pycrfsuite.Trainer(verbose=False)
        for xseq, yseq in zip(self.x_train, self.y_train):
            trainer.append(xseq, yseq)

        trainer.set_params({
            'c1': 1.0,  # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 50,  # stop earlier
            'feature.possible_transitions': True
        })

        # 把模型保存到文件中
        trainer.train(self.train_model_path)

    def predict(self):
        tagger = pycrfsuite.Tagger()
        tagger.open(self.train_model_path)

        i = self.trainset_length
        for x in self.x_test:
            result = tagger.tag(x)

            j = 0
            for r in result:
                if r == "PRODUCT":
                    print("Found product: %s" % self.tagged_instance_list[i][j][0])
                    print("Labeled as %s" % self.tagged_instance_list[i][j][len(self.tagged_instance_list[i][j]) - 1])
                j += 1
            i += 1

    def bio_classification_report(self):
        lb = LabelBinarizer()
        y_true = self.y_test

        tagger = pycrfsuite.Tagger()
        tagger.open(self.train_model_path)

        y_pred = [tagger.tag(xseq) for xseq in self.x_test]
        y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
        y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

        tagset = set(lb.classes_) - {'O'}
        tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
        class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

        return classification_report(
            y_true_combined,
            y_pred_combined,
            labels=[class_indices[cls] for cls in tagset],
            target_names=tagset,
        )


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

    # print(product_extract.tagged_instance_list[0])
    # product_extract.instance2features(product_extract.tagged_instance_list[0])

    product_extract.predict()
    print(product_extract.bio_classification_report())
