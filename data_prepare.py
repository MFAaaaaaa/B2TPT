import os
import json
import random

def create_caltech101_json(dataset_path, output_json):
    data = []

    # 获取所有类别文件夹
    categories = [category for category in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, category))]
    i = 0
    # 遍历每个类别
    for category in categories:

        category_path = os.path.join(dataset_path, category)


        # 获取该类别下的所有图像文件
        # images = [os.path.join(category_path, image) for image in os.listdir(category_path) if image.endswith('.jpg')]
        images = [os.path.join(category, image) for image in os.listdir(category_path) if image.endswith('.jpg')]


        # 随机选择一些图像作为示例，你可以根据需要更改数量
        # sample_images = random.sample(images, min(5, len(images)))

        # 为每个图像创建一个条目，包括路径和类别
        for image_path in images:
            entry = [
                image_path,
                i,
                category
            ]
            data.append(entry)
        i += 1
    # 将数据写入 JSON 文件
    with open(output_json, 'w') as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    dataset_path = "data/dslr_Gen"  # 替换为你的 Caltech 101 数据集路径
    output_json = "data/dslr_Gen/dslr.json"  # 输出的 JSON 文件名

    create_caltech101_json(dataset_path, output_json)
