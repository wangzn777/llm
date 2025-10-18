### Model Summery ###

##### CLIP (Radford et al., 2021, OpenAI) 

1. Model Name / Paper Reference
   - CLIP: Contrastive Language–Image Pretraining
   - Paper: Learning Transferable Visual Models From Natural Language Supervision, ICML 2021, OpenAI.

2. Problem & Task
   - Problem: Train a vision model that can understand images in natural language terms without task-specific labels.
   - Task: Align images and text in a shared embedding space.
   - Input → Output:
     - Input: Image + text description.
     - Output: Embedding vectors (image and text mapped into same space).

3. Input Representation
   - Image: Preprocessed to fixed size, passed through a CNN or Vision Transformer.
   - Text: Tokenized with BPE (byte-pair encoding), mapped into word embeddings, passed through a Transformer encoder.

4. Model Architecture
   - Two-tower design:
   - Image encoder: ResNet or Vision Transformer (ViT).
   - Text encoder: Transformer (12-layer).
   - Both encoders output feature vectors.
   - Features are projected into a shared embedding space with linear layers.

5. Training Objective
   - Contrastive loss (InfoNCE):
   - Given a batch of N (image, text) pairs, maximize similarity of true pairs while minimizing similarity with mismatched pairs.
   - Symmetric loss: image→text and text→image.

6. Optimization & Training Setup
   - Optimizer: Adam.
   - Dataset: ~400M image–text pairs scraped from the web.
   - Large-scale training on GPUs/TPUs.
   - No task-specific fine-tuning during pretraining.

7. Evaluation
   - Zero-shot transfer: Evaluate on 30+ datasets (classification, retrieval).
   - Metrics: Accuracy, retrieval recall.
   - Baselines: Supervised ImageNet-trained models.
   - CLIP achieved strong zero-shot performance, sometimes close to supervised models.

8. Key Contributions / Innovations
   - Introduced scalable contrastive pretraining with natural language supervision.
   - Unified image and text into a shared embedding space.
   - Demonstrated powerful zero-shot transfer: model can classify images with just label text prompts (“a photo of a dog”).
  
9. Limitations / Open Questions
   - Relies heavily on web data → dataset bias.
   - Performance still behind task-specific models in some domains.
   - Struggles with fine-grained reasoning (e.g., counting objects, compositionality).

##### FLAVA (Meta AI, 2022)

1. Model Name / Paper Reference
   - FLAVA: A Foundational Language And Vision Alignment Model
   - Paper: FLAVA: A Foundational Language And Vision Alignment Model, Singh et al., CVPR 2022.

2. Problem & Task
   - Problem: Build a single multimodal model that works well across vision-only, text-only, and vision–language tasks.
   - Task: Unified representation learning for images and text.
   - Input → Output:
     - Input: Image, text, or image–text pair.
     - Output: Embeddings or predictions depending on downstream task (classification, retrieval, VQA, etc.).

3. Input Representation
   - Image: Split into patches (like ViT), embedded into vectors.
   - Text: Tokenized (WordPiece/BPE), mapped into embeddings.
   - Both get positional encodings.

4. Model Architecture
   - Three encoders:
   - Image encoder (ViT-like).
   - Text encoder (Transformer).
   - Multimodal encoder (cross-modal Transformer that combines image + text).
   - Outputs are aligned into a shared embedding space.
   - Pretraining includes both unimodal and multimodal pathways.

5. Training Objective
   - FLAVA uses a mixture of pretraining objectives:
     - Unimodal:
     - Masked Language Modeling (MLM).
     - Masked Image Modeling (MIM).
     - Multimodal:
     - Contrastive Learning (image–text alignment, like CLIP).
     - Masked Multimodal Modeling (predict masked tokens/patches jointly).
     - Global Matching (determine if image–text pair matches).
  
6. Optimization & Training Setup
   - Pretrained on a mixture of large-scale multimodal datasets (image–text pairs, text corpora, image datasets).
   - Optimizer: AdamW.
   - Architecture ~ 86M parameters per encoder (≈200M total).
   - Designed to be more parameter-efficient than giant multimodal models.

7. Evaluation
   - Benchmarked across three categories:
	    1.	Vision-only tasks (ImageNet classification, COCO detection).
	    2.	Text-only tasks (GLUE, sentiment analysis).
	    3.	Vision–language tasks (VQA, retrieval, NLVR2).
   - Metrics: Accuracy, recall, F1, task-specific benchmarks.
   - Results: Competitive on many benchmarks, especially strong in cross-modal transfer.

8. Key Contributions / Innovations
   - Unified multimodal foundation model: can handle image-only, text-only, and multimodal tasks in one framework.
   - Combines unimodal + multimodal pretraining objectives for better generalization.
   - More efficient than prior multimodal models while remaining versatile.

9. Limitations / Open Questions
   - Performance still lags behind specialized models on some unimodal tasks.
   - Model size and pretraining cost still high.
   - Struggles with fine-grained reasoning and compositionality.
  
##### LLaVA (Liu et al., 2023)

1. Model Name / Paper Reference
   - LLaVA: Large Language and Vision Assistant
   - Paper: Visual Instruction Tuning, Liu et al., 2023.
   - Built on top of LLaMA + CLIP ViT-L/14.

2. Problem & Task
   - Problem: Extend large language models (LLMs) like LLaMA to understand images and follow multimodal instructions.
   - Task: Visual question answering, captioning, dialogue about images.
   - Input → Output:
     - Input: Image + natural language instruction.
   - Output: Natural language response grounded in the image.

3. Input Representation
   - Image: Encoded by CLIP ViT-L/14 → visual feature embeddings.
   - Text: Tokenized via LLaMA tokenizer → text embeddings.
   - Visual features are projected into the language embedding space to align with LLaMA.

4. Model Architecture
   - Two-tower design:
	    1.	Vision encoder: CLIP ViT-L/14 (frozen).
	    2.	Language model: LLaMA (pretrained, mostly frozen).
   - Projection layer: small trainable MLP that maps visual embeddings into LLaMA’s token embedding space.
   - After projection, visual tokens are concatenated with text tokens and fed into LLaMA.

5. Training Objective
   - Two-stage training:
	    1.	Feature alignment pretraining: Align image features to language embedding space using generated captions.
	    2.	Visual instruction tuning: Fine-tune with image–instruction–response datasets (e.g., GPT-4 generated multimodal instructions).
   - Objective: Standard causal LM loss (cross-entropy).

6. Optimization & Training Setup
   - Vision encoder (CLIP) is frozen; only projection layer and LLaMA are fine-tuned.
   - Pretrained on millions of image–text pairs; instruction tuning with ~150K multimodal instruction–response pairs.
   - Optimizer: AdamW, low learning rate fine-tuning.

7. Evaluation
   - Benchmarks:
     - Visual Question Answering (VQA), GQA.
     - Captioning datasets.
     - General instruction-following with images.
   - Metrics: Accuracy, BLEU, CIDEr, GPT-based evaluation for instruction following.
   - Results: Strong zero-shot and instruction-following ability, competitive with specialized VQA models.

8. Key Contributions / Innovations
   - Simple but effective architecture: frozen CLIP + frozen LLaMA + small projection layer.
   - Visual instruction tuning: Adapts LLMs to multimodal inputs with high-quality instruction data.
   - Achieves competitive multimodal reasoning without retraining the vision or language backbone.

9. Limitations / Open Questions
   - Limited to single image inputs (no video, no multi-image reasoning).
   - Dependent on the quality of instruction-tuning data (GPT-4 synthetic).
   - Struggles with fine-grained perception (e.g., OCR, counting).


##### BLIP (Bootstrapping Language-Image Pretraining, Li et al., 2022)

1. Model Name / Paper Reference
   - BLIP: Bootstrapping Language–Image Pretraining
   - Paper: BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation, Li et al., ICML 2022.

2. Problem & Task
   - Problem: Pretrain a vision–language model that works well for both understanding (e.g., VQA, retrieval) and generation (e.g., captioning).
   - Task: Unify multimodal understanding and generation with one model.
   - Input → Output:
     - Input: Image and/or text.
     - Output: Text (caption, answer) or embeddings (for retrieval).

3. Input Representation
   - Image: Processed by a Vision Transformer (ViT) pretrained on ImageNet-21k.
   - Text: Tokenized with WordPiece, embedded into vectors.
   - Both modalities fed into a Transformer-based encoder–decoder framework.

4. Model Architecture
   - Three main components:
        1.	Vision encoder: ViT backbone.
        2.	Text encoder/decoder: Transformer-based (BERT-like).
        3.	Multimodal encoder–decoder: Fuses image + text features.
   - Supports both encoder-only (understanding tasks) and encoder–decoder (generation tasks).

5. Training Objective
   - BLIP introduces three objectives:
        1.	Image–Text Contrastive (ITC): Align image and text embeddings (like CLIP).
        2.	Image–Text Matching (ITM): Classify if image–text pair matches (fine-grained alignment).
        3.	Language Modeling (LM): Generate text conditioned on image and/or text inputs (captioning, VQA).

6. Optimization & Training Setup
   - Pretraining on 129M image–text pairs (filtered from web data).
   - Dataset bootstrapping: uses the pretrained model itself to filter noisy web data and improve training set quality.
   - Optimizer: AdamW, large-batch training on GPUs.

7. Evaluation
   - Benchmarks across understanding and generation:
     - Retrieval: COCO, Flickr30K.
     - VQA: VQAv2, GQA.
     - Captioning: COCO Captions, NoCaps.
   - Metrics: Recall@K, BLEU, CIDEr, accuracy.
   - Results: BLIP achieved state-of-the-art on several multimodal benchmarks.

8. Key Contributions / Innovations
   - Unified multimodal framework that works for both understanding and generation.
   - Bootstrapped pretraining data: model improves its own dataset quality.
   - Combines contrastive, matching, and generative objectives for robust multimodal alignment.

9. Limitations / Open Questions
   - Performance tied to the quality of web-crawled data.
   - Still computationally expensive compared to unimodal models.
   - Struggles with complex reasoning and compositionality.


------------------------------
##### LLaMA (Meta AI, 2023)

1. Model Name / Paper Reference
   - LLaMA: Large Language Model Meta AI
   - Paper: LLaMA: Open and Efficient Foundation Language Models, Meta AI Research, 2023.

2. Problem & Task
   - Problem: Build an efficient, open large language model comparable to GPT-3 but trainable with fewer resources.
   - Task: General-purpose text generation and understanding (e.g., question answering, summarization, reasoning).
   - Input → Output:
     - Input: Text tokens.
     - Output: Predicted next token (autoregressive language modeling).

3. Input Representation
   - Text is tokenized using SentencePiece with Byte-Pair Encoding (BPE).
   - Tokens are mapped into embedding vectors.
   - Rotary Positional Embeddings (RoPE) used for position information instead of absolute embeddings.
  
4. Model Architecture
   - Type: Decoder-only Transformer (like GPT).
   - Key components:
   - Multi-head self-attention.
   - Feedforward networks (SwiGLU activation instead of ReLU/GeLU).
   - Pre-normalization (RMSNorm).
   - Variants: LLaMA-7B, 13B, 33B, and 65B parameters.
   - Uses smaller models but trained on more data, emphasizing efficiency.

5. Training Objective
   - Standard causal language modeling: predict next token given previous context.
   - Loss: Cross-entropy between predicted token distribution and ground truth.

6. Optimization & Training Setup
   - Trained on 1.4 trillion tokens from publicly available datasets (no proprietary data).
   - Optimizer: AdamW.
   - Data quality emphasized (filtered CommonCrawl, GitHub, Wikipedia, books, etc.).
   - Long training with large batch sizes on thousands of GPUs.

7. Evaluation
   - Benchmarked on NLP tasks: commonsense reasoning, reading comprehension, MMLU (Massive Multitask Language Understanding).
   - Metrics: Accuracy, perplexity.
   - Results:
   - LLaMA-13B outperformed GPT-3 (175B) on many benchmarks.
   - LLaMA-65B was competitive with state-of-the-art models like PaLM.

8. Key Contributions / Innovations
   - Showed that smaller models with more data can match/exceed much larger models.
   - Introduced efficiency improvements (RoPE embeddings, SwiGLU, RMSNorm).
   - Released weights (for research), democratizing access to large language models.
  
9. Limitations / Open Questions
   - Still autoregressive, inherits limitations of GPT-like models (hallucination, bias).
   - Trained on open web data → data quality and ethical concerns.
   - Not instruction-tuned by default (LLaMA-2 and LLaMA-3 later addressed this).

#####  BERT (Bidirectional Encoder Representations from Transformers, Devlin et al., 2018)

1. Model Name / Paper Reference
   - BERT: Bidirectional Encoder Representations from Transformers
   - Paper: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Devlin et al., 2018 (Google AI).

2. Problem & Task
   - Problem: Traditional NLP models (e.g., word2vec, GPT-1) either produced static embeddings or were unidirectional.
   - Task: Provide contextual, bidirectional embeddings that can be fine-tuned for a wide variety of NLP tasks.
   - Input → Output:
     - Input: Text sequence (tokens).
     - Output: Contextual embeddings (per token) or classification result (via [CLS] token).

3. Input Representation
   - Each token represented by the sum of three embeddings:
        1.	Token embedding (wordpiece).
        2.	Segment embedding (sentence A/B for pair tasks).
        3.	Position embedding (positional information).

4. Model Architecture
   - Based on Transformer encoder only (stacked).
   - Variants:
     - BERT-base: 12 layers, 12 heads, hidden size 768 (110M params).
     - BERT-large: 24 layers, 16 heads, hidden size 1024 (340M params).
   - Uses bidirectional self-attention → every token attends to both left and right context.

5. Training Objective
   - Two self-supervised pretraining tasks:
	    1.	Masked Language Modeling (MLM): Randomly mask 15% of tokens and predict them.
     	- Example: “I love [MASK]” → predict “dogs”.
	    2.	Next Sentence Prediction (NSP): Predict whether sentence B follows sentence A.
        - Helps with tasks like QA, entailment.

6. Optimization & Training Setup
   - Pretrained on BooksCorpus (800M words) + English Wikipedia (2.5B words).
   - Optimizer: Adam with warmup, large batch training.
   - Training scale: Tens of GPUs, ~1M update steps.

7. Evaluation
   - Benchmarks: GLUE, SQuAD, SWAG.
   - Metrics: Accuracy, F1, EM (Exact Match).
   - Results: Achieved state-of-the-art on 11 NLP benchmarks in 2018.

8. Key Contributions / Innovations
   - Introduced deep bidirectional pretraining with Transformers.
   - Unified pretraining + fine-tuning framework for many NLP tasks.
   - Produced contextual embeddings (word meaning changes depending on sentence).
   - Sparked the Transformer revolution in NLP.

9. Limitations / Open Questions
   - NSP objective later found unnecessary (RoBERTa dropped it).
   - Large pretraining cost, not efficient.
   - Fixed input length (512 tokens).
   - Not generative (cannot produce long text like GPT).


##### ViT (Vision Transformer, Dosovitskiy et al., 2020)
1. Model Name / Paper Reference
   - ViT: Vision Transformer
   - Paper: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, Dosovitskiy et al., 2020 (Google Research).

2. Problem & Task
   - Problem: CNNs dominated vision tasks, but they lacked the scalability and global receptive field of Transformers.
   - Task: Apply the Transformer encoder (from NLP) directly to images for classification.
   - Input → Output:
     - Input: Image.
     - Output: Class label.

3. Input Representation
   - Patch embeddings:
     - Image (e.g., 224×224) split into fixed-size patches (e.g., 16×16).
     - Flatten each patch → linear projection → patch embedding.
   - Position embeddings: Added to retain spatial information.
   - [CLS] token: Special learnable embedding prepended, used for classification.

4. Model Architecture
   - Pure Transformer encoder stack (no convolutions).
   - Components:
     - Multi-head self-attention (MHSA).
     - Feed-forward layers.
     - Layer norm + residual connections.
   - Output: [CLS] token embedding → classification head (MLP).
   - Model sizes: ViT-Base (12 layers, 768 hidden, 12 heads), ViT-Large, ViT-Huge, etc.

5. Training Objective
   - Standard supervised cross-entropy loss for classification.
   - Pretraining: on large-scale datasets (ImageNet-21k, JFT-300M).
   - Fine-tuning: on smaller datasets (ImageNet-1k).

6. Optimization & Training Setup
   - Optimizer: Adam with warmup + cosine decay.
   - Large batch sizes.
   - Data augmentation: random crop, flip, color jitter, etc.
   - Strong regularization: dropout, stochastic depth, label smoothing.

7. Evaluation
   - Benchmarks: ImageNet, CIFAR, VTAB.
   - Metrics: Top-1 / Top-5 accuracy.
   - Results:
     - Outperformed ResNets when pretrained on very large datasets.
     - With only small datasets, CNNs still generalize better.

8. Key Contributions / Innovations
   - First successful application of pure Transformers to vision.
   - Showed that scaling + large pretraining data can outperform CNNs.
   - Introduced the “patch tokenization” idea → foundation of many later models (e.g., CLIP, DINO, SAM).

9. Limitations / Open Questions
   - Data hungry: needs huge pretraining datasets.
   - Computationally expensive.
   - Lack of inductive bias (e.g., translation equivariance of CNNs).
   - Struggles on smaller datasets without special tricks.

##### DINO (Caron et al., 2021)

1. Model Name / Paper Reference
   - DINO: Self-Distillation with No Labels
   - Paper: Emerging Properties in Self-Supervised Vision Transformers, Caron et al., 2021 (Meta AI).

2. Problem & Task
   - Problem: How to train vision transformers (ViTs) without labels while still learning strong semantic features.
   - Task: Self-supervised learning of image representations that transfer well to classification, detection, and segmentation.
   - Input → Output:
     - Input: Unlabeled images.
     - Output: Semantic embeddings (feature vectors).

3. Input Representation
   - Image: Split into patches, embedded into vectors (ViT input).
   - Data augmentations: crops, color jitter, blur, solarization (different “views” of the same image).

4. Model Architecture
   - Teacher–Student framework (like knowledge distillation, but without labels):
   - Student network: Vision Transformer (ViT) updated by gradient descent.
   - Teacher network: Same architecture, updated as an exponential moving average (EMA) of student weights.
   - Both networks take different augmented views of the same image.

5. Training Objective
   - Self-distillation loss:
   - The student’s output (softmax distribution of features) is trained to match the teacher’s output on the same image (different view).
   - No labels needed — only consistency across views.
   - Additional techniques: centering + sharpening of teacher outputs to avoid collapse.

6. Optimization & Training Setup
   - Backbone: ViT-Small, ViT-Base (also ResNet for experiments).
   - Dataset: ImageNet (1.3M images, labels not used).
   - Optimizer: AdamW.
   - Tricks: heavy data augmentation, momentum teacher update.

7. Evaluation
   - Benchmarks:
     - Linear probing (freeze features, train linear classifier on ImageNet).
     - k-NN classification.
     - Transfer to object detection / segmentation.
   - Results: DINO features rival supervised pretraining.
   - Remarkable property: attention maps from ViTs align with object boundaries — without labels!

8. Key Contributions / Innovations
   - Simple self-supervised framework for ViTs without labels.
   - Introduced self-distillation (student-teacher EMA setup).
   - Showed emergent semantic segmentation capabilities from ViT attention maps.
   - Strong transfer to downstream tasks.

9. Limitations / Open Questions
   - Needs large compute (training ViTs from scratch).
   - Performance still slightly below supervised training for some tasks.
   - Sensitive to hyperparameters and augmentations.
