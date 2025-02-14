import multiprocessing
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from statistics import mean
from typing import List, Tuple
import random
import clip
import numpy as np
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import squarepad_transform, FashionIQDataset, targetpad_transform, CIRRDataset
from combiner import Combiner
from utils import extract_index_features, collate_fn, element_wise_sum, device

from data_utils import base_path, squarepad_transform, targetpad_transform, CIRRDataset, FashionIQDataset
from utils import collate_fn, update_train_running_results, set_train_bar_description, extract_index_features, \
    save_model, generate_randomized_fiq_caption, element_wise_sum, device
import json
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import clip
from utils import collate_fn, device
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 绘制混淆矩阵
def plot_confusion_matrix(conf_matrix, labels, title='Confusion Matrix', cmap='Blues'):
    """
    绘制混淆矩阵的热力图
    :param conf_matrix: 混淆矩阵
    :param labels: 类别标签
    :param title: 图的标题
    :param cmap: 热力图的颜色
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
# 测试模型
def validate_model(val_loader, clip_model, device):

    all_labels = []
    all_predictions = []
    accuracy = []
    with torch.no_grad():  # 不需要计算梯度
        val_bar = tqdm(val_loader, ncols=150)

        for idx, (reference_images, target_images, captions, labels) in enumerate(val_bar):
            # print(labels)
            captions1 = ["The wall appears thickened with brighter, dense bands and uneven, wavy contours", 
                         "Shadowing occurs near dense areas, with a coarse texture and varied echo patterns"]
            captions2 = ["The wall shows a thin, uniform layer with moderate brightness and soft contrast", 
                         "It has a smooth texture, no shadowing, and a clear boundary with the vessel lumen"]
            # captions1 = ["Large, irregular zones dominate the image, with uneven borders and rough lines",
            #              "The central space looks compressed, twisted,and surrounded by chaotic shapes"]
            # captions2 = ["Thin,uniform layers appear with clean boundaries and smooth, parallel lines",
            #              "The open area in the center is wide, rounded, and surrounded by tidy structures"]
            target_images = target_images.to(device)
            labels = labels.to(device)
                        # 编码caption1和caption2的文本
            text_inputs_1 = clip.tokenize(captions1, context_length=77, truncate=True).to(device, non_blocking=True)
            text_inputs_2 = clip.tokenize(captions2, context_length=77, truncate=True).to(device, non_blocking=True)

            # 计算目标图像的特征
            target_features, _ = clip_model.encode_image(target_images)
            target_features = target_features / target_features.norm(dim=-1, keepdim=True)  # 归一化

            # 计算两个文本描述的特征
            caption_features_1 = clip_model.encode_text(text_inputs_1)
            caption_features_1 = caption_features_1 / caption_features_1.norm(dim=-1, keepdim=True)  # 归一化

            caption_features_2 = clip_model.encode_text(text_inputs_2)
            caption_features_2 = caption_features_2 / caption_features_2.norm(dim=-1, keepdim=True)  # 归一化

            # 计算图像和文本描述之间的相似度
            similarity_1 = torch.matmul(target_features, caption_features_1.T).squeeze(0)  # 图像与caption1的相似度
            similarity_2 = torch.matmul(target_features, caption_features_2.T).squeeze(0)  # 图像与caption2的相似度

            # 对每个图像，选择与caption1或caption2的相似度更高的那个类别
            predicted = torch.argmax(torch.stack([similarity_1, similarity_2], dim=-1), dim=-1)
            # print(predicted)
            predicted_class = torch.tensor(predicted[:, 0]).clone().detach()
            # print(predicted_class)
            # 保存预测值和真实标签
            all_predictions.extend(predicted_class.cpu().numpy().ravel().tolist())
            all_labels.extend(labels.cpu().numpy().ravel().tolist())
  
            # 转换为 numpy 格式
            # 确保 predicted_classes 和 labels_numpy 是一维数组
            predicted_class1 = np.array(predicted_class.cpu()).flatten()
            labels_numpy = np.array(labels.cpu()).flatten()

            correct_predictions = np.sum(predicted_class1 == labels_numpy)  # 逐元素比较并统计 True 的数量
            accuracy1 = correct_predictions / len(labels_numpy)  # 总正确数除以样本总数
            # print(accuracy1)
            accuracy.append(accuracy1)


        # 计算混淆矩阵
        conf_matrix = confusion_matrix(all_labels, all_predictions, labels=[0, 1])
        tn, fp, fn, tp = conf_matrix.ravel()

        # 计算准确率、敏感度、特异度
        # accuracy = (tp + tn) / (tp + tn + fp + fn)
        accuracy = np.array(accuracy).mean()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # 计算其他指标
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')

        # 打印结果
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Sensitivity (Recall for positive class): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # print(all_predictions) 
        # print(all_labels)    
        # accuracy = accuracy_score(all_labels, all_predictions)
        # accuracy = np.mean(accuracy)
        # precision = precision_score(all_labels, all_predictions, average='weighted')
        # recall = recall_score(all_labels, all_predictions, average='weighted')
        # f1 = f1_score(all_labels, all_predictions, average='weighted')

    return accuracy*100, sensitivity*100, specificity*100, precision*100, recall*100, f1*100, conf_matrix


    #     # 打印并记录测试集的指标
    #     print(f"Test Results :")
    #     print(json.dumps(results_dict, indent=4))
        
    # return 

# 加载训练好的 CLIP 模型
def load_clip_model(clip_model_name, clip_model_path, device):
    clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)
    if clip_model_path:
        print('Trying to load the CLIP model from', clip_model_path)
        saved_state_dict = torch.load(clip_model_path, map_location=device)
        clip_model.load_state_dict(saved_state_dict["CLIP"])
        print('CLIP model loaded successfully')

    return clip_model, clip_preprocess

def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", default='fashionIQ', type=str, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--clip-model-name", default="ViT-B/16", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    ####  16_2024-11-16_21-22-16 最高82 没加风格扰动 16_2024-12-22_22-57-36新数据最高81
    parser.add_argument("--clip-model-path", default="D:/ultrasound/xielaoban/lab/Cimclip_cau1-bifur/models/clip_finetuned_on_fiq_ViT-B/16_2024-12-26_21-36-35/saved_models/tuned_clip_best.pt", type=Path, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str, help="Preprocess pipeline")
    parser.add_argument("--test-batch-size", default=8, type=int, help="Batch size for testing")
    parser.add_argument("--validation-frequency", default=1, type=int, help="Frequency of validation during training")
    
    args = parser.parse_args()
    # Set a fixed random seed
    set_seed(42)
    # Load the pre-trained CLIP model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, clip_preprocess = clip.load(args.clip_model_name, device=device, jit=False)
    print('Trying to load the CLIP model from', args.clip_model_path)
    saved_state_dict = torch.load(args.clip_model_path, map_location=device)
    clip_model.load_state_dict(saved_state_dict["CLIP"])
    print('CLIP model loaded successfully')

    clip_model.eval().float()
    input_dim = clip_model.visual.input_resolution
    target_ratio = 1.25
    preprocess = targetpad_transform(target_ratio, input_dim)
    print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    # Test DataLoader (assuming you already have the test dataset)
    label_mapping = {
        "thickened": 1,
        "non-thickened": 0,
        }
    relative_test_dataset = FashionIQDataset('test', ["bifur"], 'relative', preprocess, label_mapping)
    # for i in relative_train_dataset:
    #     print(i)
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=8,
                                       num_workers=0, pin_memory=False, collate_fn=collate_fn,
                                       drop_last=True, shuffle=False)
    # Assuming you are using an experiment tracking tool like Comet
    experiment = None  # Replace this with actual experiment tracking if needed

    # Run the test model

    test_results = validate_model(relative_test_loader, clip_model, device)
    # 绘制混淆矩阵
    # print(test_results[6])
    # plot_confusion_matrix(test_results[6], labels=["Non-Thickened", "Thickened"])
    # Optionally save or log the test results
    print("Test completed.")
    print(test_results)
if __name__ == '__main__':
    main()