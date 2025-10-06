# CLIP model



[CLIP](https://arxiv.org/abs/2103.00020): Provides visual encoder + text encoder, mapping both into a shared embedding space for similarity comparison. [CLIP ViT](https://huggingface.co/openai/clip-vit-large-patch14)

## [CLIP](model/CLIP.md)

 > **Summary:** CLIP (Contrastive Language–Image Pretraining) is a vision-language model trained on 400 million image–text pairs collected from the internet. It learns to associate images and their corresponding textual descriptions by jointly training an image encoder and a text encoder using a contrastive loss. This enables CLIP to perform zero-shot classification, retrieval, and other tasks by comparing the similarity between image and text embeddings, without task-specific fine-tuning. CLIP demonstrates strong generalization to a wide range of visual concepts and tasks, outperforming previous models in zero-shot settings.
 > **Advantages:**
 >
 > - Enables zero-shot learning for many vision-language tasks without task-specific training.
 > - Learns from large-scale, diverse internet data, improving generalization.
 > - Supports flexible input (any image and any text) and open-vocabulary recognition.
 > - Achieves strong performance on retrieval, classification, and transfer tasks.



## [CLIP ViT](https://huggingface.co/openai/clip-vit-large-patch14)

```python
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
```

