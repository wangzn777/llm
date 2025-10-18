# ========================
# 1. import libraries and define file paths
# ========================
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel
import requests
import seaborn as sns
import scipy.stats as stats


# dataset root directory
dataset_root = os.path.join(os.getcwd(), 'data')

# clue dataset directory
clue_dataset_dir = os.path.join(dataset_root, 'clue-dataset')
clue_train_fp = os.path.join(clue_dataset_dir, 'conceptual_train_dis.csv')
img2idxmap_fp = os.path.join(clue_dataset_dir, 'img2idxmap.json')
clue_images_dir = os.path.join(clue_dataset_dir, 'images')
# clue_val_fp = os.path.join(clue_dataset_dir, 'conceptual_val.csv')
clue_test_fp = os.path.join(clue_dataset_dir, 'conceptual_test_dis.csv')

# =========================
# 2. load img2idxmap
# =========================
with open(img2idxmap_fp, 'r') as f:
    img2idxmap = json.load(f)

# =========================
# 3. load clue train datasets
# =========================
train_df = pd.read_csv(clue_train_fp)
train_df['Congrence'] = None
train_df['Negative_Caption'] = None
train_df['Negative_Image'] = None
train_df['Auxiliary_Info'] = None   

# clue_train dataset has 1,769,509 samples
# print(f'clue_train dataset has {len(train_df)} samples')

# =========================
# 4. load images
# =========================
# fp = os.path.join(clue_images_dir, os.listdir(clue_images_dir)[0])
# image = Image.open(fp)
# image

# =========================
# 5. load CLIP model and processor(image and text preprocessor)
# =========================
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()

# get first 50 samples for testing
# df = train_df.sample(50)
# df.head()
# df= train_df

# =========================
# 7. calculate similarity score for each image-text pair
# =========================
text_samples = train_df['caption'].tolist()
images_arr = []
for url in tqdm(train_df.url):
    im_fp = os.path.join(clue_images_dir, f'{img2idxmap[url]}.jpg')
    # im = Image.open(im_fp).convert("RGB")
    images_arr.append(im_fp)

# compution of similarity score of all image-text pairs
similarity = []
similarity_img2txt = []
similarity_txt2img = []
# for i, (txt, img_fp) in tqdm(enumerate(zip(text_samples, images_arr))):
#     img = Image.open(img_fp).convert("RGB")
#     inputs = processor(text=[txt], images=[img], return_tensors="pt", padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         # Compute sim_score for image-to-text and text-to-image
#         # logits_per_image = outputs.logits_per_image  # shape: [N, M], N is number of images, M is number of texts
#         # logits_per_text = outputs.logits_per_text    # shape: [M, N]
#         # Convert sim_score to softmax probabilities
#         # sim_image_to_text = logits_per_image.softmax(dim=1)  # shape: [N, M]
#         # sim_text_to_image = logits_per_text.softmax(dim=1)   # shape: [M, N]
#         similarity.append(outputs.logits_per_image.item())
#         similarity_img2txt.append(outputs.logits_per_image.item())
#         similarity_txt2img.append(outputs.logits_per_text.item())

batch_size = 100
for i in tqdm(range(0, len(text_samples), batch_size)):
    upper_idx = min(i+batch_size, len(text_samples))
    text_batch = []
    image_batch = []
    for j in range(i, upper_idx):
        text_batch.append(text_samples[j])
        image_batch.append(Image.open(images_arr[j]).convert("RGB"))
    inputs = processor(text=text_batch, images=image_batch, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        sim = outputs.logits_per_image.diagonal()
        similarity.extend(sim.numpy().tolist())

# =========================
# 8. visualize similarity score matrix
# =========================
# plt.figure(figsize=(10, 8))
# sns.heatmap(outputs.logits_per_text.numpy(), cmap='viridis')
# plt.title('Text-to-Image Similarity Heatmap')
# plt.xlabel('Text Index')
# plt.ylabel('Image Index')
# plt.grid(True)
# plt.show()

# =========================
# 9. compute 95% confidence interval for the diagonal elements of logits_per_text
# =========================
# diag = outputs.logits_per_text.diagonal()
# Convert diag to numpy array if it's a tensor
similarity_np = np.array(similarity)

alpha = 0.05
mean = np.mean(similarity_np)
std = np.std(similarity_np, ddof=1)
n = len(similarity_np)
se = std / np.sqrt(n)

t_score = stats.t.ppf(1 - alpha/2, df=n-1)
ci_lower = mean - t_score * se
ci_upper = mean + t_score * se
mask_lower = similarity_np > ci_lower
# mask_upper = np.where(similarity_np < ci_upper)
# data = similarity_np[mask_lower|mask_upper]
data = similarity_np[mask_lower]

train_df_filtered = train_df[mask_lower]
# update all congrance label
train_df_filtered["Congrence"] = True

# Plot the diagonal values with one-sided 95% confidence intervals(lower)
# plt.plot(similarity_np, label='diag')
# plt.axhline(ci_upper, color='green', linestyle='--', label='95% lower CI')
# plt.fill_between(range(n), mean, ci_lower, color='yellow', alpha=0.2, label='95% lower CI region')
# plt.legend()
# plt.title('Diagonal values with One-sided 95% Confidence Interval (lower)')
# plt.show()

# =========================
# 10. find incongruent image-text pairs based on confidence interval
# =========================
negative_caption = train_df_filtered.caption.to_list()
negative_images_fp = [os.path.join(clue_images_dir, f'{img2idxmap[u]}.jpg') for u in train_df_filtered.url.to_list()]
negative_images = [Image.open(j).convert("RGB") for j in negative_images_fp]
for i, (txt, img_fp) in tqdm(enumerate(zip(text_samples[mask_lower], images_arr[mask_lower]))):
    # similarity for positive image and negative text pair
    img = Image.open(img_fp).convert("RGB")
    im_negative_txt_inputs = processor(text=negative_caption, images=[img], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**im_negative_txt_inputs)
        negative_caption_idx = outputs.logits_per_image.argmin(dim=1)  # For each image, get the index of the text with the smallest logit
        if outputs.logits_per_image[0, negative_caption_idx] < ci_lower:
            # update negative caption
            train_df_filtered.at[i, 'Negative_Caption'] = negative_caption[negative_caption_idx]

    # similarity for negative image and positive text pair
    txt_negative_im_inputs = processor(text=[txt], images=negative_images, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**txt_negative_im_inputs)
        negative_image_idx = outputs.logits_per_text.argmin(dim=1)  # For each image, get the index of the text with the smallest logit
        if outputs.logits_per_text[0, negative_image_idx] < ci_lower:
            # update negative caption
            train_df_filtered.at[i, 'Negative_Image'] = images_arr[negative_image_idx]

train_df_filtered.to_csv(os.path.join(clue_dataset_dir, 'congrence-cohrence-train-dataset.csv'), index=False)

# Question:
# 置信区间用在哪了？最后生成的数据还是不太懂。
# 数据生成部分解释。
# 1. negatives are always the lowest ones
# 2. negatives总是采样于低于一个确定threshold的items
# 3.看一下Ananet的dataset是否对congruence- incongruence有帮助