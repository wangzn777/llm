## One-stage model:

### Preprocessing 

```mermaid
graph TD
  image[Input image] --> image_encoder(Image Encoder)
  text[Input text] --> text_embed(Text Embed)
  image_encoder --> model[Generative Model]
  text_embed --> model[Generative Model]
  model --> auxi_info{Axiliary information: object, actions, scene, situation,...}
  auxi_info --> cong_label[Congruent target label]
```

> 1. Basic idea for this part is to classify congruence and incongruence by generating auxiliary informations.
> 2. Generative Model: used for generation of auxiliary informations
> 3. Prossible model reference:
>   1. [Pham et al.2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Pham_Composing_Object_Relations_and_Attributes_for_Image-Text_Matching_CVPR_2024_paper.pdf)
>   2. [Cheng et al.2022](https://dl.acm.org/doi/10.1145/3499027)
>   3. [Wang et al. 2019](https://arxiv.org/pdf/1910.05134)
>   4. [Xu et al.2020](https://ieeexplore.ieee.org/document/8994196)



### Congruence classification

```mermaid
graph TD
  image[Input image] --> image_encoder(Image Encoder)
  text[Input text] --> text_encoder(Text Encoder)
  image_encoder --> discourse_classifier[Discourse_Classifier]
  text_encoder --> discourse_classifier[Discourse_Classifier]
  discourse_classifier --> cong_class{Congruence Relation: yes/no}
  cong_class --> loss[BCEWithLogitsLoss + TripletLoss]
```
> This part is used for congruence relation classification. 



### Coherence classification

```mermaid
graph TD
  image[Input image] --> image_encoder(Image Encoder)
  text[Input text] --> text_encoder(Text Encoder)
  image_encoder --> discourse_classifier[Discourse_Classifier]
  text_encoder --> discourse_classifier[Discourse_Classifier]
  discourse_classifier --> coh_class{Coherence Relation: r1, r2, r3, ...}
  coh_class --> loss[BCEWithLogitsLoss + TripletLoss]
```
> This part is used for coherence relation classification. 


## Two-stage model:

```mermaid
graph TD
  image1[Input image] --> image_encoder1(Image Encoder)
  text1[Input text] --> text_encoder1(Text Encoder)
  image_encoder1 --> discourse_classifier1[Discourse_Classifier]
  text_encoder1 --> discourse_classifier1[Discourse_Classifier]
  discourse_classifier1 --> auxi{Axiliary information: object, actions, scene, situation,...}
  auxi --> cong_class{Congruence Relation: yes/no}
  cong_class --> loss[BCEWithLogitsLoss + TripletLoss]


  image2[Input image] --> image_encoder2(Image Encoder)
  text2[Input text] --> text_encoder2(Text Encoder)
  image_encoder2 --> cnn[Representation Model]
  text_encoder2 --> cnn[Representation Model]
  auxi --> |concantate| cnn[Representation Model]
  cnn --> discourse_classifier2[Discourse_Classifier]
  discourse_classifier2 --> coh_class{Coherence Relation: r1, r2, r3, ...}
  coh_class --> loss[BCEWithLogitsLoss + TripletLoss]
```


```chart
- Complement:
1. Image_encoder:  Attention based model/ViT(alternative Resnet model) + Batch_norm + Fully Connection
2. Text_encoder: Sentence_encoder + Batch_norm + Fully Connection
3. Sentence_encoder: embed_layer + Bert(alternative LSTM ) [+ Attention_layer]
4. Attention_layer: hidden layer
5. Discourse_classifier: Fully Connection
6. CNN/Representation model(feature extraction)

```
