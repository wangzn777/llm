# Meeting Summery on 15.08

**Participants: Jun.-Prof. Dr. Silberer, Ziyin Wang**
**Research Topic: Classification of Text-Image Relations**

## Tasks

This meeting focus on how to classify congruence/incongruence relation, including dataset creation, negative samples generation, similarity computation between visual and text information, auxiliary infomations extraction and model chioces.

### Two Options for Dataset Generation Model

1. Unsupervised - Similarity Calculation
   1. Compute cosine similarity between image and text embeddings.
   2. Might also include auxiliary embeddings (objects/actions).
   3. Apply a threshold → above = congruent, below = incongruent.
2. Supervised - Generated Neagative Samples
   1. Create negative image–text pairs (by mismatching) in the dataset.
   2. Train a model with both positive and negative pairs.
   3. Optimize with triplet/contrastive loss to make positives closer, negatives far from embedding space.
   4. Then use this space for congruence prediction.

### Dataset Creation with Congruence Labels

1. Build dataset of positive(congruent) and negative(incongruent) image–text pairs.
   1. Positive pairs = original aligned image + text (e.g., original image + corresponding text).
   2. Negative pairs = created by mismatching (e.g., same text paired with a wrong image, or same image paired with unrelated text).
2. Explicit Congruence Defination
   Problem: How to define "incongruent"?
   1. Simple mismatching case with only different objects(e.g., image "pan" + text "pot")
      → obviously incongruent
   2. Complex mismatching case with states/actions changing(e.g., image "onion" & text "cut onion", image "boy" & text "Tom")
      → coreference between text entities and image objects?
      → No
      → Incongruent
      => text and image share the same entities (even with state differences)
      → congruence
3. Negative Samples Creation
   1. False Negative Sample:
      Origin positive pairs p_1: Image_1 "cut potato" + Text_1 "potato"
      Mismatching negative pairs p_2: Image_2 "boiled potato" + Text_1 "potato"
      => model classify p_2 as "incongruent", but they have same entities "potato", so it should be "congruent"
      → false negative sample
      → In oder to avoiding "false negative" samples, need to compute the similarity scores to do filtering.
   2. Similarity computation(Text-Text based & Text-Image based):
      1. Measurement:
         1. [CLIPScore](s)
         2. VQAScore
            1. [repo](https://github.com/linzhiqiu/t2v_metrics)
            2. [paper](https://arxiv.org/pdf/2406.13743>)
      2. Steps:
         1. Random sample negative pairs:
            $Image_m$ as $i_m$, $Text_n$ as $t_n$, $i = 1,2,\dots$
            e.g., ($i_1$, $t_2$), ($i_2$, $t_1$)
         2. Calculate $sim(t_1,t_2)$, $sim(t_1,i_2)$
         3. Average the similarity $s_1$
         4. Compute similarity of positive pairs: $sim(i_1,t_1)$, $sim(i_2, t_2)$
         5. Average the similarity $s_2$
         6. Combine s1 with $s_2$ to set a cutoff threshold
         7. Below the threshold: negative(incongruent)

### 3. Model Choice

1. Vision-Language models to encode images and texts:
   - CLIP: Provides visual encoder + text encoder, mapping both into a shared embedding space for similarity comparison. [CLIP ViT](https://huggingface.co/openai/clip-vit-large-patch14)
   - Flava: [Flava](https://huggingface.co/facebook/flava-full)
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

1. Object detection (detect if the mentioned objects in text exist in image).
2. Caption generation + comparison (generate image captions, then compare with text).
3. Action detection (what’s happening in the image vs what’s described in text).
   → These can be integrated in two ways: 1. As auxiliary information for dataset construction (e.g., better negatives). 2. As additional model input features for classification.
4. For models of extracting auxiliary informations (actions, objects, captions)
   → huggingface for Computer Vision tasks.

### 5. Experimental Strategies

For combining encoders and auxiliary information:

1. Use CLIP similarity directly for classification.
2. As auxiliary features (objects/actions/captions) into classifier.
3. Train a combined model (image encoder + text encoder + auxiliary informations).
4. Use caption–text comparison as auxiliary task for congruence classification.

### 6. Working Steps

1. Build dataset
   → Choose CLIP as baseline
   → Decide how to fuse auxiliary informations (objects, actions, captions).
2. Design different (in)congruence & (incoherence) relation classification model and experiment with auxiliary informations.
3. Meanwhile, draft a short proposal (introduction, related work, method, experiments).
