# Meeting Summery_0815

**Participants: Jun.-Prof. Dr. Silberer, Ziyin Wang**
**Research Topic: Classification of Text-Image Relations**

## topics

This meeting focus on how to classify congruence relation, including dataset creation, negative samples generation, similarity computation between visual and text information, auxiliary infomations extraction and model chioces.

### Dataset Generation

1. auxiliary inforamtion extraction: 使用模型 ViT, object detection, Yolo
   1. input: image
   2. auxiliary information:
      1. object detection
      2. action detection
      3. caption generation
2. image
3. text
4. $\star$使用模型生成 congrence, noncongrence 标签
   1. 选择模型
      1. loss
      2. optimizer
   2. 确认模型输入：image-text, auxiliary information
   3. 模型输出：label
      1. 怎么生成 congrence, noncongrence label
         1. 是否考虑语义相似（基于 auxiliary information）
            1. 例如 object 是否出现在 text 中
            2. 生成的 caption 和 text 是否相似
            3. etc
         2. 相近则为 congrence
         3. 相反则为 noncongrence
         4. 标准:
            1. [CLIPScore](s)
               > The CLIP model (Contrastive Language–Image Pretraining) maps both images and texts into a shared embedding space. It uses a visual encoder for images and a text encoder for texts, allowing direct comparison of their embeddings. This enables tasks like image–text similarity, retrieval, and classification by measuring how closely an image and text are aligned semantically.
            2. VQAScore
               1. [repo](https://github.com/linzhiqiu/t2v_metrics)
               2. [paper](https://arxiv.org/pdf/2406.13743>)
   4. 数据集
      1. data: image, text, auxiliary information
      2. label

### relation classification

1. one stage
   1. binary classification: congrence/noncongrence classifier
      1. model:
      2. input: image + text, auxiliary information
      3. output: label
      4. loss: ...
      5. optimizer
   2. multi-label classification: cohrence classifier
      1. model:
      2. input: image + text, auxiliary information
      3. output: multi-label relations
      4. loss
      5. optimizer
2. two stage

### Two Options for Congruence Classification

1. Unsupervised - Similarity Calculation
   1. Compute cosine similarity between image and text embeddings.
      **QA**
      1. Explanation: How to define the embedding space/model? Then the computation of embedding is reasonable (怎么定义 embedding 的结果，才能使得相似性计算才有意义。即如何定义 embedding space，才能解释。)
      2. The similarity computation between embeddincg output of image embedding and text embedding models
      3. How to find a properly threshold as similarity? paper reference?
      4. Which optimizer? 最小二乘法？还是高斯牛顿法？
      5. 哪些有可训练参数，哪些不包含可训练参数？
      6. 确定 loss function，optimizer
   2. Might also include auxiliary embeddings (objects/actions).
      **QA**
      1. 有哪些信息可以作为 auxiliary information?
         1. 可以参考文献、数据集，例如参考 CLUE 中的六种信息作为 auxiliary information 的模版
         2. 使用 representation model 的方法来生成 auxiliary inforamtion
   3. Apply a threshold → above = congruent, below = incongruent.
1. Supervised - Generated Neagative Samples
   1. Create negative image–text pairs (by mismatching) in the dataset.
      1. 怎么创建？随机配对 (image-caption pair)
         1. auxiliary information exatraction: 图片 -> objects (需要通过模型提取出 auxiliary information)
         2. find object in captions
            1. 确定规则：用什么模型（model for dataset genrated）判断 auxiliary information 和 caption 是否相近
            2. 根据结果来确认是否为 congrence
         3. 一条数据包含
            - data: image, caption, auxiliary informations
            - label: congrence/noncongrence
      2. 该步骤训练的模型仅作为数据集生成模型
   2. Train a model with both positive and negative pairs.
      1. 分类模型
         1. 需要根据 image-text pair + auxiliary information 来做 embedding
         2. 把 embedding 的结果放入 image encoder, text encoder, auxiliary information encoder
         3. 把 encoder 的结果进一步经过 fc (fully connected layer) 来转化为相同模式
         4. 把三个 latent space 进行 concate
         5. 把 concate 结果放入新的 classifier 进行分类
         6. 计算正负 loss
         7. 反向传播优化梯度
   3. Optimize with triplet/contrastive loss to push positives closer, negatives far from embedding space.
      1. 怎么定义 loss？可以参考 triplet losss
   4. Then use this space for congruence prediction.

### Dataset Creation with Congruence Labels(based on CITE++)

1. Build dataset of positive(congruent) and negative(incongruent) image–text pairs.
   1. Positive pairs = original aligned image + text (e.g., original image + corresponding text).
   2. Negative pairs = created by mismatching (e.g., same text paired with a wrong image, or same image paired with unrelated text).
2. Explicit Congruence Defination
   Problem: How to define "incongruent"?
   1. Simple mismatching case with only different objects(e.g., image "pan" + text "pot")
      -> obviously incongruent
   2. Complex mismatching case with states/actions changing(e.g., image "onion" & text "cut onion", image "boy" & text "Tom")
      -> coreference between text entities and image objects?
      -> No
      -> Incongruent
      => text and image share the same entities (even with state differences) -> congruence
3. Negative Samples Creation
   1. False Negative Sample:
      - Origin positive pairs p_1: Image_1 "cut potato" + Text_1 "potato"
      - Mismatching negative pairs p_2: Image_2 "boiled potato" + Text_1 "potato"
        => model classify p_2 as "incongruent", but they have same entities "potato", so it should be "congruent"
        -> false negative sample
        -> In oder to avoiding "false negative" samples, need to compute the similarity scores to do filtering.
        min fn upper boundary = max tn lower boundary
   2. Similarity computation(Text-Text based & Text-Image based):
      1. Measurement:
         1. [CLIPScore](s)
         2. VQAScore
            1. [repo](https://github.com/linzhiqiu/t2v_metrics)
            2. [paper](https://arxiv.org/pdf/2406.13743>)
      2. Steps:
         1. Random sample negative pairs: e.g., $I_1^+(i_1) + T_2^-(t_2), \quad I_2^-(i_2) + T_1^+(t_1)$
         2. Calculate $sim(t_1,t_2)$, $sim(t_1,i_2)$
         3. Average the similarity $s_1$
         4. Compute similarity of positive pairs: $sim(i_1,t_1)$, $sim(i_2, t_2)$
         5. Average the similarity $s_2$
         6. Combine s1 with $s_2$ to set a cutoff negative threshold
         7. => Below the threshold: true negative

### 3. Model Choice

1. Vision-Language models to encode images and texts:
   - [CLIP](https://arxiv.org/abs/2103.00020): Provides visual encoder + text encoder, mapping both into a shared embedding space for similarity comparison. [CLIP ViT](https://huggingface.co/openai/clip-vit-large-patch14)
     [CLIP models](model/CLIP.md)
   - Flava: [Flava](https://huggingface.co/facebook/flava-full)
     [Flava models](model/FLAVA.md)
     > **Summary:** Flava is a multimodal transformer model designed to handle both vision and language tasks, as well as their combination. It is trained on large-scale image, text, and image-text data, enabling it to perform a wide range of tasks including image classification, text classification, image-text retrieval, and multimodal reasoning. Flava features separate unimodal encoders for images and text, as well as a multimodal encoder for joint understanding, allowing flexible and effective learning across modalities. The model demonstrates strong performance on both unimodal and multimodal benchmarks, highlighting its versatility and generalization ability.
   - Llava: [Llava 1.6](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)
   - [Llava 1.5](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
   - Blip: [Blip](https://huggingface.co/Salesforce/blip2-flan-t5-xl)
2. Other possible encoders:
   - Text: BERT, llama.
   - Visual: [DINO](https://huggingface.co/facebook/dinov2-large)
   - Multimodal: LLaVA, FLAVA, instruction-tuned models.
3. Strategy:
   Start with CLIP as the baseline.
   Compare with alternatives if time/resources allow.

### 4. Auxiliary Information Extraction

1. Object detection (detect if the mentioned objects in text exist in image). if the detected object(image patch, bbox, object-label) appeared in text
2. Action detection (what’s happening in the image vs what’s described in text).
   - These can be integrated in two ways:
     - As auxiliary information for dataset construction (e.g., better negatives).
     - As additional model input features for classification.
3. Comparsion: generated caption + orignal text(caption) (generate image captions, then compare with text).
   1. generated caption 依赖 object, action, etc. $C = \sum_i^n{w_i \cdot attr_i} = W \cdot A$
   <!-- > These can serve as features for congruence classification. -->
4. For models of extracting auxiliary informations (actions, objects, captions)
   -> huggingface for Computer Vision tasks.

### 5. Experimental Strategies

For combining encoders and auxiliary information:

- Use CLIP similarity directly for classification.
- As auxiliary features (objects/actions/captions) into classifier.
- Train a combined model (image encoder + text encoder + auxiliary info).
- Use caption–text comparison as auxiliary task for congruence classification.

### 6. Working Steps

1. build dataset
   → choose CLIP as baseline
   → decide how to fuse auxiliary info (objects, actions, captions).
2. Meanwhile, draft a short proposal (intro, related work, method, experiments).
