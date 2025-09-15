Flava: [Flava](https://huggingface.co/facebook/flava-full)

**Summary:** Flava is a multimodal transformer model designed to handle both vision and language tasks, as well as their combination. It is trained on large-scale image, text, and image-text data, enabling it to perform a wide range of tasks including image classification, text classification, image-text retrieval, and multimodal reasoning. Flava features separate unimodal encoders for images and text, as well as a multimodal encoder for joint understanding, allowing flexible and effective learning across modalities. The model demonstrates strong performance on both unimodal and multimodal benchmarks, highlighting its versatility and generalization ability.

**FlavaModel Architecture (Graph Structure):**

```mermaid
graph TD
    A[FlavaModel]
    A --> B[Text Model]
    A --> C[Image Model]
    A --> D[Multimodal Model]
    B --> B1[Text Embeddings]
    B --> B2[Text Encoder 12 layers]
    B2 --> B3[Text Pooler]
    C --> C1[Patch Embeddings]
    C --> C2[Image Encoder 12 layers]
    C2 --> C3[Image Pooler]
    D --> D1[Multimodal Encoder 6 layers]
    D1 --> D2[Multimodal Pooler]
    A --> E[Projection Heads]
    E --> E1[Image Projection]
    E --> E2[Text Projection]
    E --> E3[Image to MM Projection]
    E --> E4[Text to MM Projection]
```

```model
FlavaModel(
  (text_model): FlavaTextModel(
    (embeddings): FlavaTextEmbeddings(
      (word_embeddings): Embedding(30522, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.0, inplace=False)
    )
    (encoder): FlavaEncoder(
      (layer): ModuleList(
        (0-11): 12 x FlavaLayer(
          (attention): FlavaAttention(
            (attention): FlavaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (output): FlavaSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (intermediate): FlavaIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): FlavaOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (layernorm_before): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (layernorm_after): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        )
      )
    )
    (layernorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (pooler): FlavaPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (image_model): FlavaImageModel(
    (embeddings): FlavaImageEmbeddings(
      (patch_embeddings): PatchEmbeddings(
        (projection): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
      )
      (dropout): Dropout(p=0.0, inplace=False)
    )
    (encoder): FlavaEncoder(
      (layer): ModuleList(
        (0-11): 12 x FlavaLayer(
          (attention): FlavaAttention(
            (attention): FlavaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (output): FlavaSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (intermediate): FlavaIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): FlavaOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (layernorm_before): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (layernorm_after): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        )
      )
    )
    (layernorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (pooler): FlavaPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (multimodal_model): FlavaMultimodalModel(
    (encoder): FlavaEncoder(
      (layer): ModuleList(
        (0-5): 6 x FlavaLayer(
          (attention): FlavaAttention(
            (attention): FlavaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (output): FlavaSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (intermediate): FlavaIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): FlavaOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (layernorm_before): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (layernorm_after): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        )
      )
    )
    (layernorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (pooler): FlavaPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (image_projection): Linear(in_features=768, out_features=768, bias=True)
  (text_projection): Linear(in_features=768, out_features=768, bias=True)
  (image_to_mm_projection): Linear(in_features=768, out_features=768, bias=True)
  (text_to_mm_projection): Linear(in_features=768, out_features=768, bias=True)
)
```
