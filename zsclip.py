import torch
import clip
from PIL import Image
import os

def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = clip.load("ViT-B/16", device=device)
    return model, transform, device

def predict_image(model, transform, image_path, class_names):
    image = transform(Image.open(image_path)).unsqueeze(0).to(device)
    text_tensor = clip.tokenize(class_names).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tensor)

    # 计算图像和文本之间的相似性得分
    similarity_scores = (text_features @ image_features.T).squeeze().tolist()

    return similarity_scores

def test_images_in_folder(model, transform, root_folder_path):
    correct_predictions = 0
    total_images = 0

    for category_folder in os.listdir(root_folder_path):
        category_path = os.path.join(root_folder_path, category_folder)

        if os.path.isdir(category_path):
            # 获取类别名称
            category = category_folder

            # 初始化该类别的正确预测数
            correct_predictions_category = 0
            total_images_category = 0


            # 遍历该类别下的每个图像
            for filename in os.listdir(category_path):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    total_images += 1
                    total_images_category += 1

                    image_path = os.path.join(category_path, filename)

                    # 使用类别名称进行测试
                    similarity_scores = predict_image(model, transform, image_path, category)

                    # 找到最相似的类别
                    _,predicted_category_index = max(similarity_scores)
                    predicted_category = category_folder  # 使用子文件夹的名称作为预测的类别

                    # print(f"Image: {filename}, Predicted Class: {predicted_category}, Similarity Scores: {similarity_scores}")

                    # 如果预测类别与实际类别相同，则认为预测正确
                    if predicted_category == category:
                        correct_predictions += 1
                        correct_predictions_category += 1

            accuracy_category = correct_predictions_category / total_images_category if total_images_category > 0 else 0
            print(f"\nAccuracy for Category {category}: {accuracy_category * 100:.2f}% (Correct Predictions: {correct_predictions_category}/{total_images_category})\n")

    accuracy = correct_predictions / total_images if total_images > 0 else 0
    print(f"\nAverage Accuracy: {accuracy * 100:.2f}% (Correct Predictions: {correct_predictions}/{total_images})")

if __name__ == "__main__":
    # 加载 CLIP 模型
    clip_model, clip_transform, device = load_clip_model()

    # 定义包含图像的根文件夹路径
    root_folder_path = "/home/fameng/project/BPT-VLM/data/Art_Gen/image_data"

    # 进行测试并输出准确率
    test_images_in_folder(clip_model, clip_transform, root_folder_path)
