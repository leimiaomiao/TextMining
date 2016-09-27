import xml.etree.ElementTree as ET


def extract_main_info(doc):
    root = ET.fromstring(doc)
    info = root.find(".//AJJBQK[@nameCN='案件基本情况']")
    if info is not None:
        main_info = info.get("value")
        return main_info
    else:
        return ""
