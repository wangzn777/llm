# Classification of text / image relations

## **Topic:**

*Cross-modal* *relation classification*: Given an *image-text pair*,

1. $\star$ predict if the image and text are
   1. **congruent** (i.e., literal match -- their meaning literally match or overlap, the literal elements can be (partially) aligned or correspond to each other; no contradiction, no misalignment)?
   2. **coherent** (i.e., make sense together; the connection may be indirect or metaphorical, the meaning may be abstract, implied or complementary)
   > 具体类别需要看论文，根据别人的文章来：
   > 1. 可以是现成的数据集（类别、种类已经已知、已确定）
   >    1. 为什么这个数据集符合你的要求？ image + text pair / relation label: pipeline
   >       1. image + text pair
   >       2. relation label
   >       3. model: **baseline** ? evaluation: good / worse, compare
   >          1. how to find a baseline? model of the paper
   >          obj. generate relation between image and text pair.
   > 2. 基于相似、相同的数据集有哪些文章，使用了哪些模型?
   >     1. 搜谁引用了？
   >     2. 解决的问题：代表解决的是相似或者相近的问题
2. if image and text are coherent, classify their relationship using discourse relations (aka coherence relation; e.g., elaboration, illustration, causal, temporal)

### **Tasks, the focus can vary depending on the interest of the student:**

- **Develop models** that learn the **classification task**
  - use the model as baseline: code? contact author?
- **Evaluate and analyse** the model effectiveness
  - metrices, loss
- **Develop an XAI** approach that explains the relationship, e.g., using local methods (attention on image regions or words)

### **Data:**

- CITE (alikhani2019cite)
- General-domain data (e.g., conceptual captions CC12M) and/or social media posts (see the references below)

### **Possible RQs:**

- Which **auxiliary information** is helpful for **relationship classification**?
  E.g., domain (procedures vs. general situations vs. social media), object labels, scene labels etc.
  > **做一个新的classification的模型**，基于这个新的模型能更准确找到合适的image-text的relation
  >2-stage：1阶段检测是否congruent（image-text matching task），2阶段检测是哪些coherence relation（cross-modal coherence model）
  >对数据的处理，image encoder和text encoder（取决于选input中哪些feature，e.g.，object detection- image patch- object label）（single-steam/dual-stream architecture）
  >
  >1. 先做一个包含/考虑所有因素的分类模型作为baseline 2.筛选重要信息（auxiliary information）优化原模型结构

- Which **visual or linguistic factors**, respectively, are especially challenging for the classification models?
  For example, implicit information in one or both of the modalities, ambiguity, imperatives, etc.
  >看论文找到visual/linguistic factors，
  >medr，定量分析，定性分析，confusion matrix（看visual/linguistic factors对关系的影响）
  >不同visual/linguistic factor在不同指标（recall， F- Score等/其他论文中的指标）下对结果的影响
  >
  >
  >评估不同visual/linguistic factor的指标，分析其对所有模型的影响

- What in **the image / text establishes** coherence?
  Possible hypotheses: **actions**, **objects**, **effects** (i.e., the situation)
  >参考论文
  >
  >
  >
  >
  >
  >

### References

#### Relationships

- Taxonomy of computable cross-modal relations that characterise the contextual/literal and semiotic/logico-semantic relation between text and images of social media posts or advertisements
  - social media posts or advertisements: vempala2019categorizing, kruk2019integrating, zhang2018equal
  - general-domain data: otto2020characterization
- Discourse / coherence relations
  - in the domain of cooking recipes: alikhani2019cite
  - evaluating (V)LLMs on relation detection: ramakrishnan2025cordial
- Interaction response (i.e., prediction-specific)
  - task prediction depends on one modality (uniqueness), either modality (redundancy), and both modalities (synergy): liang2023multimodal

#### Applications

- relation-aware methods for captioning (alikhani2020cross) and image retrieval (alikhani2022cross)
- affect recognition: zhang2024camel, xiao2024vanessa

#### Representation learning

- https://arxiv.org/pdf/2502.16282
- vempala2019categorizing: https://aclanthology.org/P19-1272.pdf
- kruk2019integrating: https://aclanthology.org/D19-1469/
- zhang2018equal: http://bmvc2018.org/contents/papers/0228.pdf
- otto2020characterization: https://link.springer.com/article/10.1007/s13735-019-00187-6
- alikhani2019cite: https://aclanthology.org/N19-1056/
- (see also https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1048874/full)
- ramakrishnan2025cordial: https://arxiv.org/pdf/2502.11300
- liang2023multimodal: https://doi.org/10.1145/3577190.3614151
- alikhani2020cross: https://aclanthology.org/2020.acl-main.583/
- alikhani2022cross: https://ojs.aaai.org/index.php/AAAI/article/view/21285
- zhang2024camel: https://dl.acm.org/doi/10.1609/aaai.v38i8.28787
- xiao2024vanessa: https://aclanthology.org/2024.findings-emnlp.671/
