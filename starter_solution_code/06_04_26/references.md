# Thursday, June 4: Field Data Collection & Model Architecture

Today we focus on collecting our own real field data and implementing a self-supervised model architecture using the pre-trained DINOv2 model.

---

##  Morning Session: Field Data Collection (Slot 1)

Before you head out to the fields to collect images with your mobile devices, review these data collection guidelines:

### Best Practices for Agricultural Datasets
1.  **Angle Diversity**: Take pictures from the top of the canopy, side angles, and underneath leaves. Disease spots are not always visible from the top!
2.  **Lighting Conditions**: Capture images in full sun, overcast conditions, and under shadows. Models can fail if they only see leaves in perfect, bright daylight.
3.  **Clear Focus**: Blur is a major source of noise. Make sure the lens is focused on the leaf surface, not on the background soil or weeds.
4.  **Class Balance**: Try to collect an even distribution of classes:
    *   *DicambaDamage*
    *   *FrogEyeLeafSpot*
    *   *GenericFeeding*
    *   *InsectDamage*
    *   *Soybeans* (Healthy)
    *   *SuddenDeathSyndrome*
5.  **Scale / Reference**: Include images with various levels of disease severity (e.g., light spots vs. heavily damaged leaves).

---

##  Afternoon Session: DINOv2 & Classification Heads

### Self-Supervised Learning (SSL)
Traditionally, models are trained from scratch using millions of labeled images. In this project, we use **Transfer Learning** with DINOv2, a model pre-trained by Meta using self-supervised learning on 142 million diverse images. 
*   **Frozen Backbone**: Since DINOv2 already knows how to detect textures, shapes, edges, and leaf structures, we "freeze" the backbone weights so we do not overwrite its general knowledge.
*   **Classifier Head**: We add a single new trainable fully connected layer (`nn.Linear`) at the end of the backbone. During training, we only optimize this linear head to map DINOv2 features to our specific soybean classes.

### Dimension Alignment
The DINOv2 model architecture we are using is `vit_base_patch14_dinov2.lvd142m` (a Vision Transformer). 
*   **Backbone Features**: The output of the backbone is a feature vector of size `768`.
*   **Input Size**: Images must be resized to `(518, 518)` to match the patch dimensions of this specific ViT model.
*   **Linear Mapping**: Our linear layer will map from `768` input features to `num_classes` (6 outputs representing our soybean classes).

```text
Input Leaf Image (518x518) 
      │
      ▼
DINOv2 Backbone (Frozen) 
      │
      ▼ (768 features)
nn.Linear Classification Head (Trainable)
      │
      ▼ (6 logits)
Softmax / Class Probabilities (e.g. FrogEyeLeafSpot)
```

---

##  Curated Resources for Day 2
*   [DINOv2 Official Paper (Meta AI)](https://arxiv.org/abs/2304.07193)
*   [TIMM Model Registry documentation](https://huggingface.co/docs/timm/index)
*   [PyTorch Linear Layer API Reference](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
