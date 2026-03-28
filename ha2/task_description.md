## Objective

For this assignment, you will apply the CLIP model to a dataset selected from the list provided at CLIP data prompts, excluding CIFAR10 and CIFAR100. You will utilize one of the CLIP models specified in the CLIP model script to perform zero-shot prediction on your chosen dataset.

## Assignment Details

1. Dataset and Model Selection

   1. Choose a dataset from the list provided in the CLIP data prompts URL.
   2. Do not choose CIFAR10 or CIFAR100. 
   3. Select one available CLIP model for your zero-shot prediction task.
2. Prompt Template Implementation
   Implement zero-shot prediction using two different sets of prompt templates:
   1. Simple Template: Use a straightforward template for each class, formatted as: "a photo of a {CLASS}."
   2. Ensemble Template: Use a combination of multiple prompt templates for each class as suggested in the CLIP data prompts documentation. There is more than one way to implement the use of multiple prompt templates in CLIP zero-shot prediction. You are encouraged to explore an efficient implementation approach, although the efficiency of your implementation will not influence your grade.

3. Reporting
   Write a simple report for submission. Your report should include:
   1. Summary: Begin with a clear statement of the chosen dataset and CLIP model.
   2. Methodology: Describe the prompt templates used and explain your implementation strategy for employing multiple prompt templates per class.
   3. Results: Provide a comparative analysis of the accuracy achieved with each set of prompt templates. Highlight key findings and discuss any patterns observed.
   4. Visualization: Include visualizations of several example predictions, especially cases where misclassification occurred. These visualizations may mimic the style shown in the CLIP tutorial.

   The report must be self-contained, clear, and complete without supplementary code, as the evaluation focuses solely on the quality and completeness of the report.

## Submission

- Submit a report in PDF format via Canvas.
- File name format: Student_ID_HA2.pdf

## Academic Integrity

All submissions must follow the course policy on collaboration and citation. If you use external code, libraries, or resources beyond standard frameworks, clearly cite them in your report. 