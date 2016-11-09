import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup


def extract_main_info(doc):
    try:
        root = ET.fromstring(doc)
        info = root.find(".//AJJBQK[@nameCN='案件基本情况']")
        if info is not None:
            main_info = info.get("value")
            return main_info
        else:
            return ""
    except ET.ParseError:
        return ""


def extract_ah_info(doc):
    try:
        root = ET.fromstring(doc)
        info = root.find(".//AH[@nameCN='案号']")
        if info is not None:
            main_info = info.get("value")
            return main_info
        else:
            return ""
    except ET.ParseError:
        return ""


def extract_accuser_clarify_info(doc):
    root = ET.fromstring(doc)
    info = root.find(".//YGSCD[@nameCN='原告诉称段']")
    if info is not None:
        main_info = info.get("value")
        return main_info
    else:
        return ""


def extract_word_from_xml(doc):
    word_list = []

    soup = BeautifulSoup(doc, "lxml")
    element_list = soup.find_all("word")
    for element in element_list:
        word_list.append(element["cont"])
    return word_list
