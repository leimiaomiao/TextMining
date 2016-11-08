import requests
import config.config as config


def word_segment(sentence="我是中国人。"):
    url = 'http://api.ltp-cloud.com/analysis/?api_key=%s&text=%s&pattern=ws&format=plain' % (
    config.api_key, str(sentence))

    response = requests.post(url, None, headers={"Content-Type": "text/plain;charset=utf-8"})

    if response.ok:
        return response.content.decode("utf-8")
    else:
        return ""

