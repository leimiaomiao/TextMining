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
