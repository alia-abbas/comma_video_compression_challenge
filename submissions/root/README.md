The strategy is now a **Hybrid Pipeline**: INT4 for the vision backbone (stability) and PolarQuant for the conditioning (compression).

### **The Updated File Structure**

```text
comma_video_compression_challenge/
├── ... 
└── submissions/
    └── root_010/
        ├── compress.sh      # Environment setup (3090-only). Installs k-quants or custom ops.
        ├── compress.py      # The Overfitter. Target: 0.mkv. Implements Muon + Asymmetric Loss + QAT.
        ├── inflate.sh       # Evaluator entry point.
        ├── inflate.py       # The Hybrid Decoder. INT4 vision weights + Polar Pose conditioning.
        └── archive.zip      # Payload: [Brotli(INT4_Weights + Polar_Pose_Deltas + Mask_Bitstream)].
```

---

### **The Revised Execution List**

**1. Pull Code (The Skeleton)**
Merge `unified_brotli` and `quantizr`. You specifically need the **FiLM conditioning layers** and the **Brotli multi-stream concatenation** logic.

**2. Clean Training Script (The Overfitter)**
`compress.py` is a closed-loop optimizer for `0.mkv`. 
*   **Goal:** Overfit until `segnet_distortion` is < 0.0005. 
*   **Method:** Instead of training a general model, you are training the **weights themselves** to be the representation of the video.

**3. Replace Absolute Temporal Embedding with RoPE**
*   **Where:** Find the `TimeEmbedding` or `TemporalPositionalEncoding` class.
*   **The Fix:** Inject **Rotary Positional Embeddings**. This ensures that as the video progresses, the model understands frame $N$ is related to frame $N-1$ via a rotation in feature space, not just a static ID. This stabilizes the **PoseNet** score.

**4. Replace AdamW with Muon Optimizer**
*   **The Switch:** Drop the `Muon` class into `compress.py`. 
*   **Benefit:** Muon is better at maintaining the internal rank of your weights. This is critical when you eventually squeeze them into 4-bit; it prevents the "collapsed" layers that cause SegNet to misclassify road as grass.

**5. Apply Asymmetric Loss Masking**
*   **The Fix:** Inside the loop: `segnet_loss = SegNet(original, recon)`. 
*   **The Mask:** `weight_mask = (segnet_labels == ROAD_CAR_CLASSES) ? 5.0 : 0.1`.
*   **Operation:** `final_loss = (segnet_loss * weight_mask).mean()`. This forces the model to ignore sky/trees and use its limited INT4 capacity on the driving path.

**6. Weight Tying (Parameter Efficiency)**
*   **The Fix:** Use a single shared backbone for processing the mask and the pose conditioning. In `inflate.py`, the same `Conv2d` blocks should handle the feature refinement for both temporal dynamics and semantic reconstruction.

**7. Revised Quantization (The Hybrid Squeeze)**
*   **PolarQuant (For Pose/Conditioning):** Convert $(x, y)$ velocity vectors to $(r, \theta)$. Store as `int8` angles. This is "lossless" for PoseNet but takes half the space of `fp16` Cartesian coords.
*   **INT4 + QAT (For Vision Weights):** Do not go to 1-bit yet. Use **Quantization-Aware Training** to simulate 4-bit rounding during the training of `compress.py`. Save the backbone as `int4`.
*   **The Decoder:** `inflate.py` loads the 4-bit weights, dequantizes them to `fp16` in VRAM, and runs the forward pass.

---

### **Technical Guardrail**
If `segnet_distortion` doesn't drop below 0.001 after 20 minutes of training on the 3090, your **Asymmetric Mask** is too aggressive or your **INT4** scale factors are drifting. Adjust the mask ratio from `5.0/0.1` to `2.0/0.5` to stabilize.