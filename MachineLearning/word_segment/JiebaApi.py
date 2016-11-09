import jieba
import jieba.posseg as pseg


def word_segment(content):
    seg_list = jieba.cut(content, cut_all=False, HMM=True)
    return seg_list


def pos_tokenize(content):
    words = pseg.cut(content)

    for word, flag in words:
        print("%s %s" % (word, flag))


if __name__ == "__main__":
    seg = word_segment("给原告销售'红樱子'有机高粱种子")
    print(" ".join(seg))

    pos_tokenize("给原告销售'红樱子'有机高粱种子")
