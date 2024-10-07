# fine-tuning-a-LayoutLMv3-model-for-Named-Entity-Recognition-NER
# Handwritten Text Recognition with LayoutLMv3

This project utilizes the LayoutLMv3 model for handwritten text recognition, specifically targeting the extraction of key information from scanned documents.

## Important Parts

### 1. Environment Setup

*   Mounts Google Drive for data access.
*   Installs necessary libraries, including `transformers`, `datasets`, `torch`, `imgaug`, `seqeval`, `accelerate`, `pdf2image`, `pymupdf`, `albumentations`, `evaluate`, and `optuna`.
*   Imports required modules from these libraries.

### 2. Data Loading

*   `read_annotation_file`: Reads JSON annotation files containing bounding box and label data.
*   `display_image_with_ner_tags`: Displays an image with NER tags overlaid on bounding boxes.

### 3. Data Augmentation

*   `get_train_transform`: Defines data augmentation strategies, including median blur, color jitter, padding, and rotation.

### 4. LayoutLMv3

*   `label2id` and `id2label`: Define label mappings for token classification.
*   `processor`: Initializes a LayoutLMv3Processor for image and text preprocessing.
*   `CustomDataset`: Custom dataset class for loading images, text, and bounding box annotations. It includes data augmentation and preprocessing using the LayoutLMv3Processor.
*   `custom_collate_fn`: Collates data into batches for training, handling padding and attention masks.

### 5. Model and Metrics

*   `model`: Initializes a LayoutLMv3ForTokenClassification model for NER.
*   `compute_metrics`: Defines evaluation metrics using the `seqeval` library, calculating precision, recall, F1, and accuracy.

### 6. Training

*   `training_args`: Specifies training hyperparameters using `TrainingArguments`.
*   `MLflowTrainer`: A custom Trainer class that logs metrics using MLflow.
*   `class_weights_tensor`: Used for class weighting during training.
*   Training loop: Uses the Trainer to fine-tune the LayoutLMv3 model with data augmentation, hyperparameter optimization, and early stopping.
