import jieba


def word_segment(content):
    seg_list = jieba.cut(content, cut_all=False, HMM=True)
    return seg_list


if __name__ == "__main__":
    seg = word_segment("给原告销售'红樱子'有机高粱种子")
    print(" ".join(seg))
