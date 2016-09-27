import jieba


def word_segment(content):
    seg_list = jieba.cut(content, cut_all=False, HMM=True)
    return seg_list


if __name__ == "__main__":
    seg = word_segment("我爱北京天安门.")
    print(" ".join(seg))
