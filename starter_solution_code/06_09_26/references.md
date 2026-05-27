# Tuesday, June 9: Inference Pipeline & Actionable Insights

Today we develop our inference pipeline to make predictions on individual raw images and map those predictions to agricultural recommendations for farmers.

---

##  Morning Session: Single Image Inference

During training, we processed batches of images. In deployment (inference), we process a single image at a time. The pipeline steps are:

1.  **Load Checkpoint**: Load the saved `.pth` file containing the weights using `torch.load()`.
2.  **Model Prep**: Initialize the model structure, load the weights, and put it in evaluation mode (`model.eval()`).
3.  **Image Preprocessing**:
    *   Load image using PIL and ensure it is RGB mode.
    *   Resize image to `(518, 518)` and convert to tensor.
    *   **Dimension Expansion**: PyTorch models expect input shape `(Batch_Size, Channels, Height, Width)`. For a single image, this is `(1, 3, 518, 518)`. We add this dummy batch dimension using `img_tensor.unsqueeze(0)`.
4.  **Forward Pass**: Feed the tensor through the model to get raw logit outputs.
5.  **Softmax**: Apply `torch.softmax` to transform logits into confidence probability scores between 0 and 1.
6.  **Argmax**: Use `torch.argmax` to pick the class index with the highest probability.

---

##  Afternoon Session: Actionable Treatment Mapping

AI models are only useful if they provide actionable value. We map the predicted crop diseases to farmer-friendly treatment protocols:

| Predicted Class | Agricultural Condition | Actionable Treatment Recommendation |
| :--- | :--- | :--- |
| `Soybeans` | Healthy Soybean leaf | Crop is healthy. No treatment needed. Maintain standard watering and weed control. |
| `FrogEyeLeafSpot` | Fungal Leaf Spot | Fungal spot detected. Monitor spot density. If it exceeds 10% of canopy, apply a strobilurin fungicide. |
| `SuddenDeathSyndrome` | Soil-borne fungal disease | Soil-borne fungus (Fusarium). In-season fungicides are ineffective. Improve field drainage and plan crop rotation for next season. |
| `DicambaDamage` | Chemical herbicide drift damage | Herbicide drift detected. Document weather/wind records, inspect spray buffer zones, and coordinate with neighboring farms. |
| `InsectDamage` / `GenericFeeding` | Pest foliage feeding damage | Active insect feeding detected. Deploy sticky traps to monitor pest levels. If damage spreads, apply targeted organic or chemical pesticide. |

---

##  Curated Resources for Day 5
*   [Saving and Loading PyTorch Models Tutorial](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
*   [PyTorch Tensor unsqueeze API Reference](https://pytorch.org/docs/stable/generated/torch.unsqueeze.html)
*   [OSU Extension Soybean Diseases Field Guide](https://extensionpubs.osu.edu/)
