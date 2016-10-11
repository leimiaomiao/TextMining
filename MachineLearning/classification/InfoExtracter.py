from util.FileReader import read_accuser_clarify_info
import glob
import csv


def prepare_sample(source_dir_path, target_file_path="sample.csv", sample_num=500):
    file_list = glob.glob(source_dir_path)

    rows = []
    count = 1
    for file in file_list:
        if count <= sample_num:
            content = read_accuser_clarify_info(file, extract=True, decode="utf-8")
            if len(content.lstrip()) > 0:
                rows.append([count, content, ""])
                count += 1
        else:
            break
    print(len(rows))

    with open(target_file_path, "w", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'YGSCD', 'Name'])
        writer.writerows(rows)


if __name__ == "__main__":
    prepare_sample("../公开民事文书xml/*.xml")
