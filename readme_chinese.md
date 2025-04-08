
---
Project: Fine Tuning T5 for Paper Abstraction

Author: CHOU CHIA-HSUAN

Date: 2025-04-18

Course: 生成式人工智慧

---


## 1. Fine tuning 過程概述

1. 使用 **google/flan-t5-base** 模型進行全文摘要任務的 **Full Fine-Tuning**，藉由設計 prompt，並針對論文的 introduction 欄位生成精確的 abstract，搭配評估指標（ROUGE 與 BERTScore），挑選表現最佳的模型。
2. 使用 LoRA 微調 flan-t5-xl
3. 結合 Vamsi/T5_Paraphrase_Paw 改寫資料，來增加資料集，進行 flan-t5-base full fine tuning。但結果都沒有 full fine tuning 好，所以我最後還是用**full fine tuning**作為提交模型。


## 2. 資料

| 資料集 | 筆數 | 百分比 |
|--------|------|--------|
| 訓練集 | 367  | 90%    |
| 驗證集 | 41   | 10%    |

除了訓練與驗證資料外，另有測試資料共 103 筆，用於最終摘要結果的評估。


## 3. 三種方法參數選取和結果比較：

# 1. **google/flan-t5-base** 模型進行全文摘要任務的 **Full Fine-Tuning**

* Model：google/flan-t5-base
* 參數量：247,577,856

* Prompt：

    "You are a professional academic summarizer. "

    "Write a precise and objective abstract for the following research introduction. "

    "Do not include poetic or exaggerated language. "

    "Only describe the main objectives, methods, and key findings of the paper. "

    "If the text contains formulas, mathematical notations, or specific numerical results, retain them in the abstract. "

    "Do not add personal opinions or restate this prompt. Use a formal academic tone.\n\n"

    Introduction: {introduction}

* 訓練參數
  * learning_rate=2.5e-5
  * num_train_epochs=20
  * max_input_length = 2048
  * max_target_length = 600
  * per_device_train_batch_size=1
  * per_device_eval_batch_size=1
  * label_smoothing_factor=0.1
  * warmup_steps=500
  * lr_scheduler_type="linear"

* 測試參數
  * max_new_tokens= 600
  * min_length=200
  * num_beams=4             
  * early_stopping=True    
  * repetition_penalty=1.2     
  * no_repeat_ngram_size=3  

# 2.  使用 LORA finetuning google/flan-t5-xl

為了驗證更大模型是否有助於提升摘要品質，我嘗試使用更大的模型**flan-t5-xl進行fine tuning**，並且搭配**LORA**使用。

  * Model：google/flan-t5-xl
  * 參數量：2,868,631,552

  * Prompt：同上
  * LORA
    * r=16,                            
    * lora_alpha=32,                   
    * target_modules=["q", "v", "k", "o"], 
    * lora_dropout=0.1,  
    * trainable params: 18,874,368(0.6580%)
  * 訓練參數
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

  * 測試參數
    * max_new_tokens= 600
    * min_length=200
    * num_beams=4             
    * early_stopping=True    
    * repetition_penalty=1.2     
    * no_repeat_ngram_size=3  

# 3.使用data augmentation+Full fine tuning google/flan-t5-base

在Full fine tuning flan-t5-base的過程中，我在想可能因為訓練資料只有367筆，因此表現較差，所以對訓練資料進行**data augmentation**來提升提升模型的泛化能力。

* 流程：
  1. 使用**Vamsi/T5_Paraphrase_Paw**來對訓練集進行改寫，解碼出新的、語意相同但詞彙結構不同的 paraphrased introduction。
  2. 隨機選出 50% 的 paraphrase 結果（避免太多改寫破壞資料分布）(202筆)
  3. 用 BERTScore 過濾 paraphrased 結果的品質，保留 bert score F1 score> 0.85 的句子。(197筆)
  4. 把過濾後的 paraphrased 資料加回原始資料中，並且 90 % 為訓練集、10% 為驗證集。(605筆，訓練544筆/驗證161筆)
  5. 採用與 baseline 相同的 prompt 與訓練參數進行 full fine-tuning，但表現略差於 Full Fine-Tuning T5。




## 4. 三種模型的比較

* 以下用驗證集的表現比較

| 模型設定 | 方法 |ROUGH-1 | ROUGH-2 | ROUGH-L | Bert Score(F1) |
|--------|------|--------|--------|------   |--------|
|flan-t5-base |Prompt + Full Fine Tuning| 0.4861|0.1613|0.2471  |0.8694|
|flan-t5-xl + LORA |Prompt + LORA Fine Tuning|0.4381| 0.1284|0.2112 |0.8391|
|flan-t5-base + augmentation	Full Fine Tuning  |data aug + Prompt + Fine Tuning| 0.4509|0.1455|0.2321 |0.8637|


* Final test data Result 

| 模型設定 | 方法 |ROUGH-1 | ROUGH-2 | ROUGH-L | Bert Score(F1) |
|--------|------|--------|--------|------   |--------      |
|flan-t5-base |Prompt + Full Fine Tuning|0.4738|0.1593|0.2446 |0.8615|

(baseline 0.47/0.12/0.22/0.85)


* 觀察與推論：
  1. flan-t5-xl + LORA 較 flan-t5-base Full Fine Tuning 表現差的原因：
     1. LoRA 模組調整有限：
    目前僅針對 attention 層中的 q, k, v, o 模組插入 LoRA，可能因為微調較小所以表現較差。我之後可以考慮擴大插入模組的範圍，例如加入 FFN 中間層（如 "wi", "wo"），或嘗試更大規模的模型，如 google/flan-t5-xxl（11B）

  2. flan-t5-base + augmentation	Full Fine Tuning 較 flan-t5-base Full Fine Tuning 表現差的原因：
    1. paraphrased 資料導致訓練資料偏誤，因為 ROUGE 是考量字詞重疊的指標。即使 paraphrased text 保有語意一致，若用字或語序與 ground truth 差異過大，可能導致模型生成時傾向使用非典型詞彙，進而降低與參考摘要的字面重疊程度，造成 ROUGE 分數下滑。

    2. T5 在低資源任務中已能有效泛化，paraphrasing 對表現提升有限。

