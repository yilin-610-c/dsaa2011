# THE HONG KONG UNIVERSITY OF SCIENCE & TECHNOLOGY (GUANGZHOU) 
## Deep Learning | HA1: Reproducing ResNet on FashionMNIST-Resplit 

**Due Date:** See canvas.

---

### Instructions 
* **Format:** Your answers should be typed, not handwritten.
* **File Type:** You may submit PDF only. 
* **Quality:** The PDF must be a text-based (searchable) PDF, not an image-only/scanned PDF. Image-based PDFs will not be accepted.
* **Platform:** Submissions must be made via Canvas.
* **Dataset:** Please note that the dataset is sourced from Lab 3.

**Copyright Statement:** The materials provided by the instructor in this course are for the use of the students enrolled in the course. Copyrighted course materials may not be further disseminated.

---

### Objective 
Reproduce the core idea of ResNet (convolutional neural network with residual connections) and evaluate it on the FashionMNIST-Resplit dataset. You will implement the model, run experiments, and write a concise technical report.

---

### Tasks 

#### 1. Implement ResNet and at least 3-4 model variants.
* **Core Components:** Implement the main ResNet building blocks, including the convolutional stem, residual blocks, skip connections, and normalization such as BatchNorm.
* **Comparison:** Reuse, reproduce, or customize different model structures and compare their classification performance.
* **Model Set Requirements:** 
    * At least one **plain (vanilla) model** without residual connections.
    * At least **three ResNet-style variants** with different depth, width, or training capacity (e.g., resnet-x10, resnet-x20).
* **Documentation:** Clearly describe each variant in your report, including changes in depth, blocks, channels, normalization, or downsampling strategy.
* **Framework:** You may choose any reasonable training pipeline (e.g., PyTorch, transformers.Trainer).

#### 2. Run experiments on FashionMNIST-Resplit.
* **Evaluation:** Train all variants on the provided split and report results on the same evaluation split.
* **Control Variables:** Keep settings consistent across models (data preprocessing, optimizer, learning rate schedule, epochs, batch size, augmentations).
* **Reporting:** 
    * Final test accuracy.
    * Key training details (optimizer, learning rate, weight decay, etc.).
    * Model capacity indicators (e.g., parameter count).

#### 3. Write a Technical Report.
Your report must cover:
* **(a) Implementation:** Details on the architecture and how it matches the paper.
* **(b) Experimental results:** Quantitative results presented via tables or figures.
* **(c) Discussion and conclusions:** Interpretation of observations with evidence.
* **(d) Additional notes (optional):** Extra ablations, visualizations, or troubleshooting.

---

### Required Discussion and Grading Focus 
* **Analysis:** Discuss empirical findings, such as differences in predictions and their potential causes.
* **Criteria:** Grading is based on work volume, coherence of results, insightfulness of discussions, and clarity.

---

### Submission & Academic Integrity 
* **Submission:** A PDF technical report.
* **Policy:** Follow course policy on collaboration and citation.
* **Citations:** Clearly cite any external code or libraries used beyond standard frameworks.