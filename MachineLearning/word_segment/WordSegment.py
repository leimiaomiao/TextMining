import jieba


def word_segment(content):
    seg_list = jieba.cut(content, cut_all=False, HMM=True)
    return seg_list

