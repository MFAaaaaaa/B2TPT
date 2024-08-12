import pandas as pd

# 数据列表
data = [0.90625, 0.90625, 0.84375, 0.90625, 0.6875, 0.96875, 0.9375, 0.84375,
        0.8125, 0.90625, 0.90625, 0.90625, 0.96875, 0.96875, 0.84375, 0.9375,
        0.96875, 0.90625, 0.875, 0.875, 0.875, 1.0, 0.9375, 0.9375, 0.875,
        0.90625, 0.9375, 0.84375, 0.90625, 0.96875, 0.90625, 0.96875, 0.90625,
        0.8125, 0.875, 0.96875, 0.90625, 0.8125, 0.96875, 0.96875, 0.875,
        0.90625, 0.90625, 0.84375, 0.90625, 0.90625, 0.875, 0.9375, 0.9375,
        0.9375, 0.875, 0.9375, 0.96875, 0.9375, 1.0, 0.90625, 0.96875, 0.96875,
        0.90625, 0.90625, 0.9375, 0.84375, 0.9375, 0.9375, 0.875, 0.84375,
        0.90625, 0.84375, 0.9375, 0.9375, 0.9375, 0.90625, 1.0, 0.875, 0.8125,
        0.90625, 0.84375, 1.0]

# 创建一个DataFrame
df = pd.DataFrame(data, columns=["Value"])

# 将DataFrame写入Excel文件
output_file = "output.xlsx"
df.to_excel(output_file, index=False)

print(f"数据已保存到 {output_file}")


# import torch
# from torch.nn import functional as F
#
# import json
# import os
# import shutil
# import numpy as np
#
# import json
#
# import os
#
# import os
# import shutil
#
# import os
# import shutil
#
# import scipy.io
# import numpy as np
#
# import scipy.io
# import pandas as pd
# import numpy as np
#
# import scipy.io
# import pandas as pd
#
# import os
# import shutil

# for i in range(2):
#     print(i)
# ee = torch.tensor([0]).cuda()
# ff = torch.tensor(0.).cuda()
# c = torch.tensor([])
# a = torch.tensor([[2, 1, 2], [2, 2, 2], [2, 1, 2], [2, 2, 2], [2, 1, 2], [2, 2, 2], [2, 1, 2], [2, 2, 2]])
# b = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
# c = torch.cat([a, c], dim=0)

# c = torch.load("/home/fameng/project/BPT-VLM/best_prompt.pt")
# print(c, c.size())
# c = torch.cat((a,b),0)
# print(c)
# print(c.size())
# d,e = c.split(8,0)
# print(d,e)

# print(type(ee),ee.size())
# print(type(ff),ff.size())
# ff = torch.tensor(0.).cuda()



# import scipy.io
#
# # 读取 mat 文件
# mat_data = scipy.io.loadmat("/home/fameng/project/BPT-VLM/data/imagenet/meta.mat")
#
# # 打印所有键
# print(mat_data.keys())

# import numpy as np
# import scipy.io

# import os
# import shutil
#
# # 定义要检查的主目录和目标目录
# main_directory = "/home/fameng/project/BPT-VLM/data/imagenet/ILSVRC2012_img_val"
# target_directory = "/home/fameng/project/BPT-VLM"
#
# # 确保目标目录存在
# if not os.path.exists(target_directory):
#     os.makedirs(target_directory)
#
# # 找出不包含 .JPEG 文件的子文件夹
# empty_folders = []
#
# # 遍历主目录下的所有子文件夹
# for root, dirs, files in os.walk(main_directory):
#     for dir_name in dirs:
#         # 获取子文件夹的路径
#         sub_folder = os.path.join(root, dir_name)
#
#         # 检查子文件夹中是否有 .JPEG 文件
#         has_jpeg = any(file.lower().endswith('.jpeg') for file in os.listdir(sub_folder))
#
#         # 如果没有 .JPEG 文件，将子文件夹路径添加到 empty_folders 列表中
#         if not has_jpeg:
#             empty_folders.append(sub_folder)
#
# # 打印所有不包含 .JPEG 文件的子文件夹，并移动到目标目录
# print("以下子文件夹不包含 .JPEG 文件并将被移动到目标目录:")
# for folder in empty_folders:
#     print(folder)
#     # 获取新路径
#     new_path = os.path.join(target_directory, os.path.basename(folder))
#     # 移动文件夹到目标目录
#     shutil.move(folder, new_path)
#
# print("文件夹移动完成。")

# # 读取文件
# with open('/home/fameng/project/BPT-VLM/data/imagenet/ILSVRC2012_validation_ground_truth.txt', 'r') as file:
#     lines = file.readlines()
#
# # 处理每一行，减去1
# new_lines = []
# for line in lines:
#     try:
#         number = int(line.strip())  # 将每一行的字符串转换为整数
#         new_number = number - 1     # 数字减1
#         new_lines.append(str(new_number) + '\n')  # 将处理后的数字转换为字符串并添加换行符
#     except ValueError:
#         # 如果行内容不是整数，则跳过
#         continue
#
# # 写回文件或另存为新文件
# with open('/home/fameng/project/BPT-VLM/data/imagenet/val_gt.txt', 'w') as file:
#     file.writelines(new_lines)
#
# print("文件处理完成，结果已保存为 'gd_modified.txt'。")


# # 定义文件路径
# label_file_path = '/home/fameng/project/BPT-VLM/data/imageNet_Label.txt'  # label.txt 文件路径
# image_folder_path = '/home/fameng/project/BPT-VLM/data/imagenet-v2/image_data'  # image 文件夹路径
#
# # 读取label.txt文件内容
# with open(label_file_path, 'r', encoding='utf-8') as file:
#     content = file.read()
#
# # 在字符串两端添加花括号以形成有效的 JSON 格式
# json_content = '{' + content.strip().rstrip(',') + '}'
#
# # 解析 JSON 内容
# data = json.loads(json_content)
#
# # 创建一个字典来存储编号和新名称的映射
# rename_map = {key: value[1] for key, value in data.items()}
#
# # 遍历 image 文件夹下的所有子文件夹
# for sub_folder in os.listdir(image_folder_path):
#     sub_folder_path = os.path.join(image_folder_path, sub_folder)
#     if os.path.isdir(sub_folder_path) and sub_folder in rename_map:
#         new_sub_folder_name = rename_map[sub_folder]
#         new_sub_folder_path = os.path.join(image_folder_path, new_sub_folder_name)
#         os.rename(sub_folder_path, new_sub_folder_path)
#         print(f'Renamed: {sub_folder_path} -> {new_sub_folder_path}')
#
# print('所有子文件夹名称已修改完成。')


# # 定义文件夹路径
# folders = ['/home/fameng/project/BPT-VLM/data/imagenet-v2/imagenetv2-matched-frequency-format-val', '/home/fameng/project/BPT-VLM/data/imagenet-v2/imagenetv2-threshold0.7-format-val', '/home/fameng/project/BPT-VLM/data/imagenet-v2/imagenetv2-threshold0.7-format-val']  # 文件夹列表
# destination_folder = folders[0]  # 合并到第一个文件夹
#
# # 遍历除了第一个文件夹以外的其他文件夹
# for src_folder in folders[1:]:
#     for sub_folder in os.listdir(src_folder):
#         src_sub_folder_path = os.path.join(src_folder, sub_folder)
#         dest_sub_folder_path = os.path.join(destination_folder, sub_folder)
#
#         # 确保目标子文件夹存在
#         if not os.path.exists(dest_sub_folder_path):
#             os.makedirs(dest_sub_folder_path)
#
#         # 遍历并复制子文件夹中的文件
#         for file_name in os.listdir(src_sub_folder_path):
#             src_file_path = os.path.join(src_sub_folder_path, file_name)
#             dest_file_path = os.path.join(dest_sub_folder_path, file_name)
#
#             # 如果目标文件已存在，可以选择覆盖或跳过
#             if os.path.exists(dest_file_path):
#                 print(f'文件已存在，跳过: {dest_file_path}')
#                 continue
#
#             shutil.move(src_file_path, dest_file_path)
#             print(f'复制文件: {src_file_path} -> {dest_file_path}')
#
# print('所有文件已合并到第一个文件夹中。')

# 定义文件夹路径和输出文件路径
# image_data_folder = '/home/fameng/project/BPT-VLM/data/imagenet_Gen/image_data'  # image_data 文件夹路径
# output_file_path = '/home/fameng/project/BPT-VLM/data/imagenet_Gen/split.json'  # 输出 JSON 文件路径
#
# # 初始化数据结构
# data = {"test": []}
#
# # 遍历 image_data 文件夹下的所有子文件夹及其文件
# for label_index, sub_folder in enumerate(sorted(os.listdir(image_data_folder))):
#     sub_folder_path = os.path.join(image_data_folder, sub_folder)
#     if os.path.isdir(sub_folder_path):
#         for image_file in sorted(os.listdir(sub_folder_path)):
#             image_file_path = os.path.join(sub_folder, image_file)
#             data["test"].append([image_file_path, label_index, sub_folder])
#
# # 将数据写入到 sqlit.json 文件中
# with open(output_file_path, 'w', encoding='utf-8') as file:
#     json.dump(data, file, indent=4)
#
# print(f'数据已处理并写入到 {output_file_path}')


# # 定义文件路径
# label_file_path = '/home/fameng/project/BPT-VLM/data/imageNet_Label.txt'  # label.txt 文件路径
#
# # 读取label.txt文件内容
# with open(label_file_path, 'r', encoding='utf-8') as file:
#     lines = file.readlines()
#
# # 初始化存储名称及其行号的字典
# name_map = {}
# duplicates = []
#
# # 遍历每一行并检查名称是否重复
# for line_number, line in enumerate(lines, start=1):
#     # 使用正则表达式提取[]中的内容
#     match = re.search(r'\["[^"]+", "([^"]+)"\]', line)
#     if match:
#         name = match.group(1)
#         if name in name_map:
#             duplicates.append((name, line_number, name_map[name]))
#         else:
#             name_map[name] = line_number
#
# # 输出重复内容所在的行数
# if duplicates:
#     print("以下名称重复：")
#     for name, current_line, previous_line in duplicates:
#         print(f'名称: "{name}" 在第 {previous_line} 行和第 {current_line} 行')
# else:
#     print("没有重复的名称。")


# # 定义文件路径
# label_file_path = '/home/fameng/project/BPT-VLM/data/imageNet_Label.txt'  # label.txt 文件路径
# root_folder_path = '/home/fameng/project/BPT-VLM/data/imagenet-sketch'  # 需要修改子文件夹名称的根文件夹路径
#
# # 读取label.txt文件内容
# with open(label_file_path, 'r', encoding='utf-8') as file:
#     content = file.read()
#
# # 在字符串两端添加花括号以形成有效的 JSON 格式
# json_content = '{' + content.strip().rstrip(',') + '}'
#
# # 解析 JSON 内容
# data = json.loads(json_content)
#
# # 创建一个字典来存储旧名称和新名称的映射
# rename_map = {value[0]: value[1] for value in data.values()}
#
# # 遍历根文件夹下的所有子文件夹
# for folder_name in os.listdir(root_folder_path):
#     folder_path = os.path.join(root_folder_path, folder_name)
#     if os.path.isdir(folder_path) and folder_name in rename_map:
#         new_folder_name = rename_map[folder_name]
#         new_folder_path = os.path.join(root_folder_path, new_folder_name)
#         os.rename(folder_path, new_folder_path)
#         print(f'Renamed: {folder_path} -> {new_folder_path}')
#
# print('所有子文件夹名称已修改完成。')


# # 定义文件路径
# input_file_path = '/home/fameng/project/BPT-VLM/data/imageNet_Label.txt'  # 输入文本文件路径
# output_file_path = '/home/fameng/project/BPT-VLM/data/output.txt'  # 输出文本文件路径
#
# # 读取文件内容
# with open(input_file_path, 'r', encoding='utf-8') as file:
#     content = file.read()
#
# # 在每个键值对之后添加换行符，不影响键值对内部的逗号
# pattern = re.compile(r'("\d+": \[.*?\]),')
# content = pattern.sub(r'\1\n', content)
#
# # 去除末尾多余的逗号和换行符
# content = content.rstrip(',\n')
#
# # 将结果写入到输出文本文件中
# with open(output_file_path, 'w', encoding='utf-8') as file:
#     file.write(content)
#
# print(f'数据已处理并写入到 {output_file_path}')

# 定义文件路径
# input_file_path = '/home/fameng/project/BPT-VLM/data/Testing_01.txt'  # 输入文件路径
# output_file_path = '/home/fameng/project/BPT-VLM/data/test_dir.txt'  # 输出文件路径
#
# # 读取文件并删除每行的前三个字符
# with open(input_file_path, 'r', encoding='utf-8') as file:
#     lines = file.readlines()
#
# # 处理每一行，删除前三个字符
# processed_lines = [line[3:] for line in lines]
#
# # 将修改后的内容写回文件或另存为新文件
# with open(output_file_path, 'w', encoding='utf-8') as file:
#     file.writelines(processed_lines)
#
# print(f'已处理文件，并将结果保存到 {output_file_path}')

# import os
#
# # 定义文件夹路径和输出文件路径
# folder_path = '/home/fameng/project/BPT-VLM/data/SUN397_Gen/image_data'  # 要读取子文件夹名称的文件夹路径
# output_file_path = '/home/fameng/project/BPT-VLM/data/SUN397_Gen/subfolders.txt'  # 输出文本文件路径
#
# # 获取子文件夹名称并按字母顺序排序
# subfolders = sorted([name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))])
#
# # 将子文件夹名称写入文本文件
# with open(output_file_path, 'w', encoding='utf-8') as file:
#     for subfolder in subfolders:
#         file.write(subfolder + '\n')
#
# print(f'子文件夹名称已按字母顺序写入到 {output_file_path}')


# import json
#
# # 定义文件路径
# label_file_path = '/home/fameng/project/BPT-VLM/data/SUN397_Gen/new_class.txt'  # 存放类别名称的文本文件路径
# output_file_path = '/home/fameng/project/BPT-VLM/data/SUN397_Gen/test_dir.txt'  # 存放图片路径的文本文件路径
# json_output_path = '/home/fameng/project/BPT-VLM/data/SUN397_Gen/split.json'  # 输出JSON文件路径
#
# # 读取类别并分配标签
# labels = {}
# with open(label_file_path, 'r', encoding='utf-8') as file:
#     for index, line in enumerate(file):
#         category = line.strip()
#         if category:
#             labels[category] = index
#
# # 初始化数据字典
# data = {"test": []}
#
# # 读取图片路径并生成JSON结构
# with open(output_file_path, 'r', encoding='utf-8') as file:
#     for line in file:
#         line = line.strip()
#         if line:
#             parts = line.split('/')
#             if len(parts) >= 2:
#                 category = parts[0]
#                 image_path = line
#                 label = labels.get(category, -1)  # 如果类别不在label文件中，分配-1作为标签
#                 data["test"].append([image_path, label, category])
#
# # 将处理后的数据写入JSON文件
# with open(json_output_path, 'w', encoding='utf-8') as json_file:
#     json.dump(data, json_file, indent=4)
#
# print(f'已将数据保存到 {json_output_path}')


# import scipy.io
#
# data = scipy.io.loadmat('/home/fameng/project/BPT-VLM/data/cars/cars_annos.mat')
# class_names = data['class_names']
# f_class = open('/home/fameng/project/BPT-VLM/data/cars/label_map.txt', 'w')
#
# num = 1
# for j in range(class_names.shape[1]):
#     class_name = str(class_names[0, j][0]).replace(' ', '_')
#     print(num, class_name)
#     f_class.write(str(num) + ' ' + class_name + '\n')
#     num = num + 1
# f_class.close()
#
# import scipy.io
#
# data = scipy.io.loadmat('/home/fameng/project/BPT-VLM/data/cars/cars_annos.mat')
# annotations = data['annotations']
# f_train = open('/home/fameng/project/BPT-VLM/data/cars/mat2txt.txt', 'w')
#
# num = 1
# for i in range(annotations.shape[1]):
#     name = str(annotations[0, i][0])[2:-2]
#     test = int(annotations[0, i][6])
#     clas = int(annotations[0, i][5])
#
#     name = str(name)
#     clas = str(clas)
#     test = str(test)
#     f_train.write(str(num) + ' ' + name + ' ' + clas + ' ' + test + '\n')
#     num = num + 1
#
# f_train.close()

# 打开原始文本文件进行读取
# with open('/home/fameng/project/BPT-VLM/data/cars/mat2txt.txt', 'r', encoding='utf-8') as infile:
#     lines = infile.readlines()
#
# # 处理每一行，删除 "car" 关键字之前的字符
# processed_lines = []
# for line in lines:
#     if 'car' in line:
#         new_line = line.split('car', 1)[1]  # 保留 "car" 及其之后的字符
#         processed_lines.append('car' + new_line)
#
# # 将处理后的行写入新的文本文件
# with open('/home/fameng/project/BPT-VLM/data/cars/output.txt', 'w', encoding='utf-8') as outfile:
#     outfile.writelines(processed_lines)
#
# print("处理完成，新文件已生成：output.txt")


# import os
#
# def replace_jpeg_with_jpg(base_folder):
#     for root, dirs, files in os.walk(base_folder):
#         for filename in files:
#             if filename.lower().endswith('.jpeg'):
#                 file_path = os.path.join(root, filename)
#                 new_file_path = os.path.join(root, filename[:-5] + '.jpg')
#                 os.rename(file_path, new_file_path)
#                 print(f"Renamed: {file_path} to {new_file_path}")
#
# # 定义基础文件夹路径
# base_folder = "/home/fameng/project/BPT-VLM/data/imagenet-r"
#
# replace_jpeg_with_jpg(base_folder)

# import os
# import json
#
# def create_image_list(base_folder, output_file):
#     data = {"train": []}
#     class_index = 0
#     class_mapping = {}
#
#     for subdir in sorted(os.listdir(base_folder)):
#         subdir_path = os.path.join(base_folder, subdir)
#         if os.path.isdir(subdir_path):
#             class_mapping[subdir] = class_index
#             for filename in sorted(os.listdir(subdir_path)):
#                 if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
#                     file_path = os.path.join(subdir, filename)
#                     data["train"].append([file_path, class_index, subdir])
#             class_index += 1
#
#     with open(output_file, 'w') as f:
#         json.dump(data, f, indent=4)
#
#     print(f"JSON file has been created: {output_file}")
#
# # 定义基础文件夹路径和输出JSON文件名
# base_folder = "/home/fameng/project/BPT-VLM/data/SUN397_Gen/image_data"
# output_file = "/home/fameng/project/BPT-VLM/data/SUN397_Gen/output.json"
#
# create_image_list(base_folder, output_file)

# import os
# import shutil
#
#
# def move_images(base_folder, txt_file):
#     with open(txt_file, 'r') as file:
#         lines = file.readlines()
#
#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
#
#         # 分割路径和文件名
#         subdir, filename = os.path.split(line)
#
#         # 创建目标子文件夹（如果不存在）
#         target_dir = os.path.join(base_folder, subdir)
#         if not os.path.exists(target_dir):
#             os.makedirs(target_dir)
#
#         # 源文件路径
#         source_file = os.path.join(base_folder, filename)
#
#         # 目标文件路径
#         target_file = os.path.join(target_dir, filename)
#
#         # 移动文件
#         if os.path.exists(source_file):
#             shutil.move(source_file, target_file)
#         else:
#             print(f"File {source_file} does not exist and cannot be moved.")
#
#
# # 定义基础文件夹路径和 all.txt 文件路径
# base_folder = "/home/fameng/project/BPT-VLM/data/aircraft/images"
# txt_file = "/home/fameng/project/BPT-VLM/data/aircraft/images/all.txt"
#
# move_images(base_folder, txt_file)


# lam = np.random.beta(0.1, 0.1)
# print(lam)
# # 定义输入和输出文件路径
# input_file = "/home/fameng/project/BPT-VLM/data/aircraft/train.txt"
# output_file = "/home/fameng/project/BPT-VLM/data/aircraft/new_train.txt"
#
# # 读取输入文件并修改每一行
# with open(input_file, "r") as f:
#     lines = [line.strip() for line in f.readlines()]
#
# # 修改每一行的内容并写入新文件
# with open(output_file, "w") as f:
#     for line in lines:
#         parts = line.split(" ", 1)
#         new_line = parts[1] + "/" + parts[0]
#         f.write(new_line + "\n")


# # 定义label.txt文件路径和images目录路径
# label_file = "/home/fameng/project/BPT-VLM/data/aircraft/variants.txt"
# images_dir = "/home/fameng/project/BPT-VLM/data/aircraft/images"
#
# # 读取label.txt文件获取类别名称
# with open(label_file, "r") as f:
#     class_names = [line.strip() for line in f.readlines()]
#
# # 在images目录下创建对应的子文件夹
# for class_name in class_names:
#     class_dir = os.path.join(images_dir, class_name)
#     if not os.path.exists(class_dir):
#         os.makedirs(class_dir)
#         print(f"Created directory: {class_dir}")
#     else:
#         print(f"Directory already exists: {class_dir}")


# input_file = "/home/fameng/project/BPT-VLM/data/aircraft/images_variant_test.txt"
# output_file = "/home/fameng/project/BPT-VLM/data/aircraft/test.txt"
#
# # 读取输入文件的每一行，并在第8个位置之后添加“.jpg”
# with open(input_file, "r") as f:
#     lines = f.readlines()
#
# modified_lines = []
# for line in lines:
#     line = line.strip()
#     modified_line = line[:7] + ".jpg" + line[7:]
#     modified_lines.append(modified_line)
#
# # 将修改后的行写入输出文件
# with open(output_file, "w") as f:
#     for line in modified_lines:
#         f.write(line + "\n")

# import json
#
# # 定义文件路径
# input_file_path = '/home/fameng/project/BPT-VLM/data/SUN397_Gen/image_data/output.txt'  # 输入文本文件路径
# output_json_path = '/home/fameng/project/BPT-VLM/data/SUN397_Gen/image_data/output.json'  # 输出JSON文件路径
#
# # 初始化数据字典
# data = {"test": []}
#
# # 读取文件并处理每一行
# with open(input_file_path, 'r', encoding='utf-8') as file:
#     for line in file:
#         line = line.strip()
#         if line:
#             parts = line.split('/')
#             if len(parts) == 2:
#                 category = parts[0]
#                 image_path = line
#                 # 这里假设标签是0，根据需要修改
#                 label = 0
#                 data["test"].append([image_path, label, category])
#
# # 将处理后的数据写入JSON文件
# with open(output_json_path, 'w', encoding='utf-8') as json_file:
#     json.dump(data, json_file, indent=4)
#
# print(f'已将数据保存到 {output_json_path}')


# # def create_json(train_file, test_file, json_file):
# def create_json(test_file, json_file):
#
#     # data = {"train": [], "test": []}
#     data = {"test": []}
#
#     class_index_map = {}
#
#     # 处理train.txt文件
#     # with open(train_file, 'r') as f:
#     #     train_data = [line.strip() for line in f.readlines()]
#     # for i, line in enumerate(train_data):
#     #     class_name, file_name = line.split('/')
#     #     if class_name not in class_index_map:
#     #         class_index_map[class_name] = len(class_index_map)
#     #     data["train"].append([line, class_index_map[class_name], class_name])
#
#     # 处理test.txt文件
#     with open(test_file, 'r') as f:
#         test_data = [line.strip() for line in f.readlines()]
#     for i, line in enumerate(test_data):
#         class_name, file_name = line.split('/')
#         if class_name not in class_index_map:
#             class_index_map[class_name] = len(class_index_map)
#         data["test"].append([line, class_index_map[class_name], class_name])
#
#     # 写入到JSON文件
#     with open(json_file, 'w') as f:
#         json.dump(data, f, indent=4)
#
#
# # train.txt和test.txt文件的路径
# # train_file = '/home/fameng/project/BPT-VLM/data/cars/new_train.txt'
# test_file = '/home/fameng/project/BPT-VLM/data/SUN397_Gen/image_data/output.txt'
#
# # 输出的JSON文件路径
# json_file = '/home/fameng/project/BPT-VLM/data/SUN397_Gen/image_data/split.json'
# #
# # # 创建JSON文件
# # create_json(train_file, test_file, json_file)
# create_json(test_file, json_file)


# def add_label_to_train(label_file, train_file, output_file):
#     # 读取label.txt中的类别名称
#     with open(label_file, 'r') as f:
#         labels = [label.strip() for label in f.readlines()]
#
#     # 读取train.txt中的训练数据
#     with open(train_file, 'r') as f:
#         train_data = [line.strip() for line in f.readlines()]
#
#     # 创建一个字典来映射train.txt中的每一行和对应的类别路径
#     train_labels = {}
#
#     # 为每一行的开头加上对应的类别名称路径，并写入到输出文件中
#     with open(output_file, 'w') as f:
#         for line in reversed(train_data):
#             # 从train.txt每一行的结尾到第一个"_"之前的数据作为类别名称
#             class_name = line.rsplit('_', 1)[0]
#             # 在label.txt中寻找与类别名称匹配的类别
#             label = next((l for l in labels if l == class_name), None)
#             if label is not None:
#                 # 将每一行及其对应的类别路径添加到字典中
#                 train_labels[line] = label
#
#         # 按照train.txt文件的顺序将每一行及其对应的类别路径写入到输出文件中
#         for line in train_data:
#             if line in train_labels:
#                 # 如果train.txt中的某一行在字典中存在对应的类别路径，则写入到输出文件中
#                 modified_line = f"{train_labels[line]}/{line}\n"
#                 f.write(modified_line)
#             else:
#                 # 如果train.txt中的某一行在字典中不存在对应的类别路径，则打印警告信息
#                 print(f"Warning: Label not found for class '{line.rsplit('_', 1)[0]}'")
#
#
#
# label_file = '/home/fameng/project/BPT-VLM/data/pets/labels.txt'
# train_file = '/home/fameng/project/BPT-VLM/data/pets/test.txt'
# output_file = '/home/fameng/project/BPT-VLM/data/pets/test_new.txt'
# # #
# # # 根据label.txt中的类别名称为train.txt文本中的每一行开头加上对应的类别名称路径，并写入到output.txt中
# add_label_to_train(label_file, train_file, output_file)

#
# def add_extension(input_file, output_file):
#     with open(input_file, 'r') as f:
#         lines = f.readlines()
#
#     with open(output_file, 'w') as f:
#         for line in lines:
#             # 将每行末尾添加 ".jpg"，然后写入新的txt文件中
#             line_with_extension = line.strip() + ".jpg\n"
#             f.write(line_with_extension)
#
# # 输入文件和输出文件的路径
# input_file = '/home/fameng/project/BPT-VLM/data/food101_Gen/meta/train.txt'
# output_file = '/home/fameng/project/BPT-VLM/data/food101_Gen/meta/trainn.txt'
# #
# # # 在txt文本每行末尾添加 ".jpg"
# add_extension(input_file, output_file)

# def remove_after_space(input_file, output_file):
#     # 读取txt文件内容，并去掉每行第一个空格之后的内容
#     with open(input_file, 'r') as f:
#         lines = f.readlines()
#
#     modified_lines = [line.split(' ', 1)[0] + '\n' for line in lines]
#
#     # 将处理后的内容写入到新的txt文件中
#     with open(output_file, 'w') as f:
#         f.writelines(modified_lines)
#
# # 输入文件和输出文件的路径
# input_file = '/home/fameng/project/BPT-VLM/data/pets/test1.txt'
# output_file = '/home/fameng/project/BPT-VLM/data/pets/test.txt'
#
# # 去除输入文件中每行结尾的最后五位字符，并保存到输出文件中
# remove_after_space(input_file, output_file)

# pets
# def create_subfolders(dataset_dir, labels_file):
#     # 创建存放子文件夹的目录
#     output_dir = os.path.join(dataset_dir, 'processed_data')
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 读取标签文件
#     with open(labels_file, 'r') as f:
#         labels = f.readlines()
#
#     # 遍历标签，创建子文件夹并移动文件
#     for label in labels:
#         # 去除换行符
#         label = label.strip()
#         # 创建子文件夹
#         class_dir = os.path.join(output_dir, label)
#         os.makedirs(class_dir, exist_ok=True)
#         # 遍历数据集文件夹中的文件
#         for file in os.listdir(dataset_dir):
#             if file.endswith('.jpg'):
#                 # 如果文件名包含当前类别的名称，则移动文件到对应的子文件夹
#                 if label in file:
#                     src_path = os.path.join(dataset_dir, file)
#                     dst_path = os.path.join(class_dir, file)
#                     shutil.move(src_path, dst_path)
#
# # 数据集目录和标签文件路径
# dataset_dir = '/home/fameng/project/BPT-VLM/data/pets/images'
# labels_file = '/home/fameng/project/BPT-VLM/data/pets/labels.txt'
#
# # 将数据集按类别名称建立子文件夹
# create_subfolders(dataset_dir, labels_file)


# def modify_json(json_file):
#     with open(json_file, 'r') as f:
#         data = json.load(f)
#
#     # 遍历每个样本
#     for item in data['test']:
#         # 获取图像文件名、标签和类别名称
#         image_filename, label, category = item
#
#         # 修改图像文件路径，加上类别名称作为前缀
#         item[0] = f"{category}/{image_filename}"
#
#     # 保存修改后的JSON文件
#     with open(json_file, 'w') as f:
#         json.dump(data, f, indent=4)
#
# # JSON文件路径
# json_file = '/home/fameng/project/BPT-VLM/data/flower102_Gen/split2.json'
#
# # 修改JSON文件中的路径格式
# modify_json(json_file)


# def create_folders_from_json(json_file, output_dir):
#     with open(json_file, 'r') as f:
#         data = json.load(f)
#
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)
#
#     for item in data['train']:
#         # 获取图像文件名、标签和类别名称
#         image_filename, label, category = item
#
#         # 创建子文件夹（如果不存在）
#         class_dir = os.path.join(output_dir, category)
#         os.makedirs(class_dir, exist_ok=True)
#
#         # 拼接图像文件路径
#         image_path = os.path.join(class_dir, image_filename)
#
#         # 在子文件夹中创建符号链接
#         # 你也可以使用shutil.copy()复制文件而不是创建符号链接
#         shutil.copy(image_filename, image_path)
#         # if not os.path.exists(image_path):
#         #     os.symlink(os.path.abspath(image_filename), image_path)
#
# # JSON文件路径
# json_file = '/home/fameng/project/BPT-VLM/data/oxford_flowers/jpg/split_zhou_OxfordFlowers.json'
#
# # 输出目录
# output_dir = '/home/fameng/project/BPT-VLM/data/oxford_flowers/image'
#
# # 根据JSON文件创建多个子文件夹
# create_folders_from_json(json_file, output_dir)


# list = [(11,2),(22,2),(33,2)]
# a = []
# for i in range(3):
#     a.append(list[i][0])
#
#
# print(list,len(list))
# print(a)


# w = torch.tensor([[0.2, 0.4, 2.0], [2.0, 0.4, 0.2]])
# # print(w.shape[0])
#
# v,i = torch.max(w,1)
# # print(v,i,v.size(),i.size())
# out = torch.tensor([[1.1, 2.2, 3.2], [3.2, 2.1, 1.5]])
# # print(out.size())
# out = F.softmax(out, dim=1)
# label = torch.tensor([2, 1])
# aaa = torch.tensor([2, 1])
# correct = (label==aaa).float().sum()
# print(correct)
# loss = F.cross_entropy(out, label, reduction='none')
# print(loss)
# # print(loss.size())
# print("******")
# los = torch.mul(F.cross_entropy(out, label, reduction='none'), v)
# print(los)

# state_dict = torch.load("/home/fameng/project/BPT-VLM/result/Visda/Visda_shallow_cma_ViT-B-16-best.pth")
# print(type(state_dict))
# for i in state_dict:
#     print(i)
#     print(type(state_dict[i]))


# print(state_dict['best_prompt_text'])


# import os#
# root_folder = "/home/fameng/DA_dataset/office_home/Art"
# class_names = [f.name for f in os.scandir(root_folder) if f.is_dir()]
# print(class_names)
# print(type(class_names))


# import torch
# import clip
# from PIL import Image
# import os
# import torch
# import torch.nn as nn
#
#
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.layer = nn.Linear(4, 2)
#
#     def forward(self, x):
#         return self.layer(x)
#
#
# def forward_hook(module, input, output):
#     print(f"Forward Hook - Input: {input}, Output: {output}")
#
#
# def backward_hook(module, grad_input, grad_output):
#     print(f"Backward Hook - Grad Input: {grad_input}, Grad Output: {grad_output}")
#
#
# # 创建模型和输入
# model = SimpleModel()
# input_data = torch.randn(1, 4)
# # 注册前向传播和反向传播的 hooks
# model.layer.register_forward_hook(forward_hook)
# model.layer.register_backward_hook(backward_hook)
# # 前向传播
# output = model(input_data)
# # Forward Hook - Input:
# # (tensor([[ 0.9603,  0.5458, -1.2675,  0.0720]]),),
# # Output: tensor([[ 1.2245, -0.0400]], grad_fn=<AddmmBackward0>)
#
# # 反向传播
# output.backward(torch.randn_like(output))
# # Backward Hook - Grad Input: (tensor([-0.0762,  1.3609]), None,
# # tensor([[-0.0732,  1.3068],[-0.0416,  0.7428],
# # [ 0.0966, -1.7250],[-0.0055,  0.0980]])),
# #  Grad Output: (tensor([[-0.0762,  1.3609]]),)
#
#
#
# #
#
# #
# # def load_clip_model():
# #     device = "cuda" if torch.cuda.is_available() else "cpu"
# #     model, transform = clip.load("ViT-B/16", device=device)
# #     return model, transform, device
# #
# #
# # def predict_image(model, transform, image_path, category):
# #     image = transform(Image.open(image_path)).unsqueeze(0).to(device)
# #     text_tensor = clip.tokenize([category]).to(device)
# #
# #     # if image.dtype != text_tensor.dtype:
# #     #     target_dtype = text_tensor.dtype
# #     #     image = image.to(dtype=target_dtype)
# #
# #     with torch.no_grad():
# #         image_features = model.encode_image(image)
# #         text_features = model.encode_text(text_tensor)
# #
# #     # 计算图像和文本之间的相似性得分
# #     similarity_score = (text_features @ image_features.T).squeeze().tolist()
# #
# #     return similarity_score
# #
# #
# # def test_images_in_folder(model, transform, root_folder_path):
# #     correct_predictions = {}
# #     total_images = 0
# #
# #     for category_folder in os.listdir(root_folder_path):
# #         category_path = os.path.join(root_folder_path, category_folder)
# #
# #         if os.path.isdir(category_path):
# #             # 获取类别名称
# #             category = category_folder
# #
# #             # 初始化该类别的正确预测数
# #             correct_predictions[category] = 0
# #
# #             # 遍历该类别下的每个图像
# #             for filename in os.listdir(category_path):
# #                 if filename.endswith(('.jpg', '.jpeg', '.png')):
# #                     total_images += 1
# #                     image_path = os.path.join(category_path, filename)
# #
# #                     # 使用类别名称进行测试
# #                     similarity_score = predict_image(model, transform, image_path, category)
# #
# #                     # 这里可以根据具体需求设置一个相似性得分的阈值
# #                     threshold = 0.5
# #                     predicted_class = "Match" if similarity_score > threshold else "No Match"
# #
# #                     print(
# #                         f"Image: {filename}, Predicted Class: {predicted_class}, Similarity Score: {similarity_score}")
# #
# #                     # 如果相似性得分超过阈值，则认为预测正确
# #                     if similarity_score > threshold:
# #                         correct_predictions[category] += 1
# #
# #     accuracy = {category: correct_predictions[category] / total_images if total_images > 0 else 0 for category in
# #                 correct_predictions}
# #     print("\nAccuracy:")
# #     for category in correct_predictions:
# #         print(
# #             f"{category}: {accuracy[category] * 100:.2f}% (Correct Predictions: {correct_predictions[category]}/{total_images})")
# #
# #
# # if __name__ == "__main__":
# #     # 加载 CLIP 模型
# #     clip_model, clip_transform, device = load_clip_model()
# #
# #     # 定义包含图像的根文件夹路径
# #     root_folder_path = "data/Art_Gen/image_data"
# #
# #     # 进行测试并输出准确率
# #     test_images_in_folder(clip_model, clip_transform, root_folder_path)
