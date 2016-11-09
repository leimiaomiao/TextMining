import requests
import re
import json


def word_segment(sentence="我是中国人。"):
    url_base = "http://127.0.0.1:12345/ltp"
    data = {
        "s": sentence,
        "x": "n",
        "t": "all"
    }
    r = requests.post(url_base, data)
    r.encoding = "utf-8"
    if r.ok:
        content = strip_xml(r.text.strip())
        return content
    else:
        return ""


def strip_xml(data):
    p = re.compile(r'<arg[^/>]+/>')
    return p.sub('', data)

if __name__ == "__main__":
    ws_ms_category_file_path = "../data/ws_ms_category.json"
    sample_list = json.load(open(ws_ms_category_file_path, "r"))

    sample_list = sample_list[1600:1684]
    for sample in sample_list:
        content = word_segment(sample["BASIC_INFO"])
        with open("../data/output/%s.xml" % sample["ID"], "wb") as file:
            file.write(content.encode("utf-8"))
        file.close()
