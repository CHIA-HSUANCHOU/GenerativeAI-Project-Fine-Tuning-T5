# GenerativeAI-Project-Fine-Tuning-T5

---
Project: Fine Tuning T5 for Paper Abstraction

Author: CHOU CHIA-HSUAN

Date: 2025-04-18

Course: Generative AI

---

# 1. Fine-Tuning Model Approaches

Goal: Generate academic abstracts from research paper introductions by fine-tuning language models. Evaluation metrics include ROUGE and BERTScore to determine the best-performing model.
Three approaches were explored:

1. Full fine-tuning of google/flan-t5-base
2. LoRA fine-tuning of google/flan-t5-xl
3. Data augmentation using Vamsi/T5_Paraphrase_Paws followed by full fine-tuning of flan-t5-base

# 2. Dataset Overview
| Data |Samples | Proportion |
|--------|------|--------|
| Train | 367  | 90%    |
|Validation | 41   | 10%    |

An additional 103 samples are used as the test set for final evaluation.

# 3. Three Method：
## Method 1: Full Fine-Tuning with google/flan-t5-base

* Model：google/flan-t5-base
* Parameters：247,577,856

* Prompt：

    "You are a professional academic summarizer. "

    "Write a precise and objective abstract for the following research introduction. "

    "Do not include poetic or exaggerated language. "

    "Only describe the main objectives, methods, and key findings of the paper. "

    "If the text contains formulas, mathematical notations, or specific numerical results, retain them in the abstract. "

    "Do not add personal opinions or restate this prompt. Use a formal academic tone.\n\n"

    Introduction: {introduction}

* Training Parameters:
  * learning_rate=2.5e-5
  * num_train_epochs=20
  * max_input_length = 2048
  * max_target_length = 600
  * per_device_train_batch_size=1
  * per_device_eval_batch_size=1
  * label_smoothing_factor=0.1
  * warmup_steps=500
  * lr_scheduler_type="linear"

* Inference Parameters:
  * max_new_tokens= 600
  * min_length=200
  * num_beams=4             
  * early_stopping=True    
  * repetition_penalty=1.2     
  * no_repeat_ngram_size=3  

## Method 2: LoRA Fine-Tuning with google/flan-t5-xl
To verify whether a larger model can improve summarization quality, I experimented with fine-tuning a larger model, flan-t5-xl, combined with LoRA.

  * Model：google/flan-t5-xl
  * Parameters：2,868,631,552
  * Prompt：Same as method one
  * LORA
    * r=16,                            
    * lora_alpha=32,                   
    * target_modules=["q", "v", "k", "o"], 
    * lora_dropout=0.1,  
    * trainable params: 18,874,368(0.6580%)
  * Training Parameters:
    * learning_rate=5e-5
    * num_train_epochs=3
    * max_input_length = 2048
    * max_target_length = 600
    * per_device_train_batch_size=1
    * per_device_eval_batch_size=1
    * label_smoothing_factor=0.1
    * warmup_ratio=0.1
    * optim="adafactor"
    * lr_scheduler_type="cosine"
  * Inference Parameters:
    * max_new_tokens= 600
    * min_length=200
    * num_beams=4             
    * early_stopping=True    
    * repetition_penalty=1.2     
    * no_repeat_ngram_size=3
    
## Method 3: Data Augmentation + Full Fine-Tuning (flan-t5-base)
To address potential data scarcity (only 367 training samples), a paraphrasing-based data augmentation strategy was applied:

* Steps:
  1. Used Vamsi/T5_Paraphrase_Paws to generate semantically equivalent but words different versions of the training introductions.
  2. Randomly selected 50% of the paraphrased examples (202 samples).
  3. Filtered paraphrases using BERTScore, retaining only those with F1 > 0.85 (197 samples).
  4. Combined filtered paraphrased data with the original dataset, resulting in 605 samples (544 training / 61 validation).
  5. Fine-tuned using the same prompt and parameters as Method 1.

## 4. Evaluation Results

* Validation Set Comparison

| Model | Method |ROUGH-1 | ROUGH-2 | ROUGH-L | Bert Score(F1) |
|--------|------|--------|--------|------   |--------|
|flan-t5-base |Prompt + Full Fine Tuning| 0.4861|0.1613|0.2471  |0.8694|
|flan-t5-xl + LORA |Prompt + LORA Fine Tuning|0.4381| 0.1284|0.2112 |0.8391|
|flan-t5-base + augmentation Full Fine Tuning  |data aug + Prompt + Fine Tuning| 0.4509|0.1455|0.2321 |0.8637|

* Final Test Set Result
* 
| Model | Method |ROUGH-1 | ROUGH-2 | ROUGH-L | Bert Score(F1) |
|--------|------|--------|--------|------   |--------      |
|flan-t5-base |Prompt + Full Fine Tuning|0.4738|0.1593|0.2446 |0.8615|

(baseline 0.47/0.12/0.22/0.85)

# 5. Observations and Discussion
1. LoRA Fine-Tuning (flan-t5-xl):

The performance of the LoRA-based fine-tuning was lower than full fine-tuning of flan-t5-base. This might be because LoRA was only applied to the attention modules (q, k, v, o). Expanding LoRA to additional modules (e.g., wi, wo in the feedforward layers) or switching to a larger model like flan-t5-xxl (11B) could improve results.

2. Data Augmentation:

Despite increasing the training data size, ROUGE scores were slightly lower.

   1. ROUGE is sensitive to words overlapping. Even if paraphrased introductions are semantically accurate, variations in wording and 
   structure may lead to lower ROUGE scores.
   2. Mismatch in style or vocabulary: The paraphrased data may introduce inconsistent writing styles or diverge from the ground truth, 
   reducing the model's alignment with target summaries.
   3. Strong generalization of T5-base: The base model already generalizes well under low-resource settings, so data augmentation did not 
   significantly improve performance.


