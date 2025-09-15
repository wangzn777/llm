[CLIP](https://arxiv.org/abs/2103.00020): Provides visual encoder + text encoder, mapping both into a shared embedding space for similarity comparison. [CLIP ViT](https://huggingface.co/openai/clip-vit-large-patch14)

- [CLIP](model/CLIP.md)
 > **Summary:** CLIP (Contrastive Language–Image Pretraining) is a vision-language model trained on 400 million image–text pairs collected from the internet. It learns to associate images and their corresponding textual descriptions by jointly training an image encoder and a text encoder using a contrastive loss. This enables CLIP to perform zero-shot classification, retrieval, and other tasks by comparing the similarity between image and text embeddings, without task-specific fine-tuning. CLIP demonstrates strong generalization to a wide range of visual concepts and tasks, outperforming previous models in zero-shot settings.
 > **Advantages:**
 >- Enables zero-shot learning for many vision-language tasks without task-specific training.
 >- Learns from large-scale, diverse internet data, improving generalization.
 >- Supports flexible input (any image and any text) and open-vocabulary recognition.
 >- Achieves strong performance on retrieval, classification, and transfer tasks.

- [CLIP ViT](https://huggingface.co/openai/clip-vit-large-patch14)
