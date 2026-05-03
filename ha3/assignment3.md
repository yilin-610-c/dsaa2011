# Deep Learning | HA3: Diffusion Model Experiments on MNIST

**University:** The Hong Kong University of Science & Technology (Guangzhou)  
**Due Date:** See web  
**Submission Platform:** Canvas  
**Submission Format:** PDF or DOCX  

---

## Instructions

- Answers should be typed, not handwritten.
- The report may be submitted in **PDF** or **DOCX** format.
- Submissions must be made via **Canvas**.
- No code submission is required.
- Similarity scores will be computed.

---

## Copyright Statement

The materials provided by the instructor in this course are for the use of the students enrolled in the course. Copyrighted course materials may not be further disseminated.

---

## Objective

In this assignment, you will extend the diffusion model implementation from the tutorial and conduct a series of experiments on the **MNIST dataset**.

The goal is to analyze how different design choices affect the quality of generated images.

---

## Assignment Details

### 1. Baseline Setup

- Start from the diffusion model implementation provided in the tutorial.
- Use **MNIST** as the dataset.

---

### 2. Experimental Variations

You are required to perform experiments by varying the following components:

#### Noise Schedule

- Try at least **two different noise schedules**.

#### Diffusion Time Steps

- Use at least **two different numbers of time steps**.

#### Sampling Method

- Compare **DDPM** and **DDIM**.

#### Requirement

- You must conduct at least **8 experiments** in total.

---

### 3. Evaluation Metrics

To evaluate the quality of generated images, consider using:

- **Inception Score (IS)**
- **Fréchet Inception Distance (FID)**

---

### 4. Additional Analysis

Diffusion models map real data to a unit Gaussian distribution in latent space.

You need to:

- Demonstrate that starting from latent vectors far from the origin leads to poor image quality.
- Use your best trained model to illustrate this phenomenon.

---

## Reporting Requirements

Your report should be well-organized and include the following sections:

### 1. Summary

Briefly describe your setup and experiment configurations.

### 2. Methodology

Explain how you varied the following factors:

- Noise schedule
- Diffusion time steps
- Sampling methods

### 3. Results

Present the results of all experiments and compare them.

### 4. Analysis

Provide insights into how different factors affect performance.

### 5. Visualization

Include generated image samples and comparisons across different settings.

---

## Submission

- Submit your report via **Canvas**.
- No code submission is required.
- Similarity scores will be computed.

---

## Academic Integrity

All submissions must follow the course policy on collaboration and citation.

Properly acknowledge any external resources used.