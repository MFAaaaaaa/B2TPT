import json
import os

import torch
from CLIPdraw import clip as clip
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from captum.attr import visualization
from dataset.general import load_train, load_test, load_fix

from CLIPdraw.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

# @title Control context expansion (number of attention layers to consider)
# @title Number of layers for image Transformer
start_layer = -1  # @param {type:"number"}

# @title Number of layers for text Transformer
start_layer_text = -1  # @param {type:"number"}


def interpret(image, texts, model, device, start_layer=start_layer, start_layer_text=start_layer_text):
    batch_size = texts.shape[0]
    # print(batch_size)
    # print(image.size())
    images = image.repeat(batch_size, 1, 1, 1)
    # print(images.size())
    # logits_per_image, logits_per_text = model(images, texts)
    logits_per_image, logits_per_text = model(images, texts)
    # print(logits_per_text)
    # print(logits_per_image)


    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

    if start_layer == -1:
        # calculate index of last layer
        start_layer = len(image_attn_blocks) - 1

    # print(image_attn_blocks[0])
    # print("**********")
    # print(image_attn_blocks[0].attn.out_proj.in_features)

    # num_tokens = image_attn_blocks[0].attn.out_proj.in_features
    # R = torch.eye(num_tokens, num_tokens).to(device)
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
            continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:]

    text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())

    if start_layer_text == -1:
        # calculate index of last layer
        start_layer_text = len(text_attn_blocks) - 1

    num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
    R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(device)
    R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(text_attn_blocks):
        if i < start_layer_text:
            continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R_text = R_text + torch.bmm(cam, R_text)
    text_relevance = R_text

    return text_relevance, image_relevance

# def preprocess_images(images, preprocess, device):
#     preprocessed_images = []
#     for img in images:
#         img = preprocess(img).unsqueeze(0).to(device)
#         preprocessed_images.append(img)
#     return torch.cat(preprocess_images)

def show_image_relevance(image_relevance, image, orig_image):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(orig_image);
    axs[0].axis('off');

    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    axs[1].imshow(vis);
    axs[1].axis('off');




# def show_heatmap_on_text(text, text_encoding, R_text):
#     CLS_idx = text_encoding.argmax(dim=-1)
#     R_text = R_text[CLS_idx, 1:CLS_idx]
#     text_scores = R_text / R_text.sum()
#     text_scores = text_scores.flatten()
#     # print(text_scores)
#     text_tokens = _tokenizer.encode(text)
#     text_tokens_decoded = [_tokenizer.decode([a]) for a in text_tokens]
#     vis_data_records = [visualization.VisualizationDataRecord(text_scores, 0, 0, 0, 0, 0, text_tokens_decoded, 1)]
#     visualization.visualize_text(vis_data_records)
def show_heatmap_on_text(text_encoding, R_text):
    CLS_idx = text_encoding.argmax(dim=-1)
    R_text = R_text[CLS_idx, 1:CLS_idx]
    text_scores = R_text / R_text.sum()
    text_scores = text_scores.flatten()
    # print(text_scores)
    # text_tokens = _tokenizer.encode(text)
    # text_tokens_decoded = [_tokenizer.decode([a]) for a in text_tokens]
    # vis_data_records = [visualization.VisualizationDataRecord(text_scores, 0, 0, 0, 0, 0, text_tokens_decoded, 1)]
    # visualization.visualize_text(vis_data_records)


# %%

clip._MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}

# %%

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device, jit=False)


def parse_batch( batch):
    image = batch["image"]
    label = batch["label"]
    # print(image.dtype)
    # print("%%%")
    image = image.to(device="cuda")
    # print(image.dtype)
    # image = image.to(device=self.device)
    label = label.to(device="cuda")

    return image, label


def load_test_images_from_json(json_path, folder_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    test_images = data['test']
    test_image_paths = [os.path.join(folder_path, img[0]) for img in test_images]
    return test_image_paths

# %%

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'



# %%


# 配置路径和参数
json_path = "/home/fameng/project/BPT-VLM/data/caltech101_Gen/split.json"
folder_path = "/home/fameng/project/BPT-VLM/data/caltech101_Gen/image_data/"
output_folder_path = "/home/fameng/project/BPT-VLM/heatmaps/"
# batch_size = 32  # 根据需求调整批次大小
# 从 JSON 文件中加载测试图像路径
test_image_paths = load_test_images_from_json(json_path, folder_path)
for img_path in test_image_paths:
    # 提取类别名称和文件名
    path_parts = img_path.split('/')
    class_name = path_parts[-2]
    file_name = path_parts[-1]
    file_name = file_name[:-4]
    img_name = class_name + file_name

    img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    texts = ["a photo of a" + class_name]
    # print(texts)
    text = clip.tokenize(texts).to(device)
    # print(text.size(), type(text))

    # 计算相关性
    R_text, R_image = interpret(model=model, image=img, texts=text, device=device)
    batch_size = text.shape[0]

    for i in range(batch_size):
        # show_heatmap_on_text(texts[i], text[i], R_text[i])
        # show_heatmap_on_text(text[i], R_text[i])

        show_image_relevance(R_image[i], img, orig_image=Image.open(img_path))

        # 保存图像到指定路径
        output_img_path = os.path.join(output_folder_path, f"heatmap_{img_name}")
        plt.savefig(output_img_path)
        plt.close()



# print(test_image_paths)
# num = len(test_image_paths) / 32
# tensor_data = torch.load("/home/fameng/project/BPT-VLM/best_prompt.pt").cpu()
# tensor_groups = tensor_data.split(8, dim=0)
# assert len(tensor_groups) == 78, "Tensor未正确切分成78组"
#
# image_groups = [test_image_paths[i:i + 32] for i in range(0, len(test_image_paths), 32)]
# assert len(image_groups) == 78, "图片路径列表未正确切分成78组"
#
# grouped_data = list(zip(image_groups, tensor_groups))
#
# for i in range(78):
#     test_image, text = grouped_data[i]
#
#
#     for img_path in test_image:
#         # 提取类别名称和文件名
#         path_parts = img_path.split('/')
#         class_name = path_parts[-2]
#         file_name = path_parts[-1]
#         file_name = file_name[:-4]
#         img_name = class_name + file_name
#
#
#         img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
#         # texts = ["a photo of a" + class_name]
#         # print(texts)
#         # text = clip.tokenize(texts).to(device)
#         # print(text.size(), type(text))
#
#         # 计算相关性
#         R_text, R_image = interpret(model=model, image=img, texts=text, device=device)
#         batch_size = text.shape[0]
#
#         for i in range(batch_size):
#             # show_heatmap_on_text(texts[i], text[i], R_text[i])
#             show_heatmap_on_text(text[i], R_text[i])
#
#             show_image_relevance(R_image[i], img, orig_image=Image.open(img_path))
#
#             # 保存图像到指定路径
#             output_img_path = os.path.join(output_folder_path, f"heatmap_{img_name}")
#             plt.savefig(output_img_path)
#             plt.close()
#
#         # print(len(test_image_paths))




