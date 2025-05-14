# Parameter-Efficient-Fine-Tuning
Adaptive Text Summarization Using Soft Prompt Tuning, LoRA, and Traditional Fine-Tuning
## Overview
This project involves comparing three different fine-tuning methods for a pre-trained GPT-Neo model on a text summarization task:

- Soft Prompt Tuning: Adapts the model using task-specific embeddings prepended to the inputs.
- LoRA (Low-Rank Adaptation): Adds low-rank matrices to adapt the model weights efficiently.
- Traditional Fine-Tuning: Updates only the final layers of the model for task-specific learning.

The goal is to determine which fine-tuning method best balances performance and resource efficiency.

## Prerequisites
Make sure the following libraries are installed:
```
Python 3.8+
PyTorch
Transformers (pip install transformers)
PEFT (pip install peft)
Pandas (pip install pandas)
Rouge Score (pip install rouge-score)
tqdm (pip install tqdm)
```
Make sure you are running the code on a system with GPU support, as training and evaluation can be resource-intensive.

## Dataset Preparation
Use the CNN/DailyMail dataset for summarization, with splits:

- Training Set: 21,000 samples
- Validation Set: 6,000 samples
- Test Set: 3,000 samples

Each sample should have the following fields:
- article: The input article to be summarized.
- highlights: The target summary.

## How to Execute the Code
- Step 1: Clone Repository and Install Dependencies
```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
```
- Step 2: Running Soft Prompt Tuning
Load Pre-trained Model and Tokenizer:
```python
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
```
- Initialize Soft Prompts: Follow the SoftPromptEmbedding class from the code to set up trainable embeddings, then proceed to training:
```python
train(model, tokenizer, soft_prompt, optimizer, train_loader, epochs=6)
```
- Evaluation: Run evaluation to generate and save predictions:
```python
evaluate(model, tokenizer, soft_prompt, test_loader, save_csv=True, csv_filename="soft_prompt_test_results.csv")
```
- Step 3: Running LoRA (Low-Rank Adaptation)
Initialize LoRA Layers:
```python
from peft import get_peft_model, LoraConfig
lora_config = LoraConfig(...)
model = get_peft_model(model, lora_config)
```
- Train:
```python
train_lora(model, train_loader, epochs=6)
```
- Evaluation:
```python
evaluate_lora(model, test_loader, save_csv=True, csv_filename="lora_test_results.csv")
```
- Step 4: Running Traditional Fine-Tuning (Last Layer Only)
Freeze Model Layers and Unfreeze the Last Layer(s):
```python
for param in model.parameters():
    param.requires_grad = False
for param in model.lm_head.parameters():
    param.requires_grad = True
```
- Train:
```python
train_finetune(model, train_loader, epochs=10)
```
- Evaluation:
```python
evaluate_finetune(model, test_loader, save_csv=True, csv_filename="finetune_test_results.csv")
```
- Saving and Restoring Models
Saving Model Weights
- Soft Prompt Embeddings:
```python
torch.save(soft_prompt.state_dict(), "soft_prompt_trained.pt")
```
- LoRA Model Weights:
```python
torch.save(model.state_dict(), "lora_trained.pt")
```
- Traditional Fine-Tuned Model Weights:
```python
torch.save(model.state_dict(), "finetune_trained.pt")
```
- Loading Saved Models
- Load Soft Prompt Embeddings:
```python
soft_prompt.load_state_dict(torch.load("soft_prompt_trained.pt"))
```
- Load LoRA Model Weights:
```python
model.load_state_dict(torch.load("lora_trained.pt"))
```
- Load Fine-Tuned Model Weights:
```python
model.load_state_dict(torch.load("finetune_trained.pt"))
```
- Calculating ROUGE Scores from Saved CSV
To compute ROUGE scores after evaluation:
```python
from rouge_score import rouge_scorer
import pandas as pd

def calculate_rouge_from_csv(csv_filename):
    df = pd.read_csv(csv_filename)
    predictions = df['Predicted Summary'].tolist()
    references = df['Actual Summary'].tolist()
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        for key in scores.keys():
            scores[key].append(score[key].fmeasure)
    
    avg_scores = {key: sum(values) / len(values) for key, values in scores.items()}
    print(f"ROUGE-1: {avg_scores['rouge1']:.4f}, ROUGE-2: {avg_scores['rouge2']:.4f}, ROUGE-L: {avg_scores['rougeL']:.4f}")
    
    return avg_scores

# Example usage:
calculate_rouge_from_csv("lora_test_results.csv")
```

## Notes and Recommendations
- GPU Memory: Ensure your system has enough GPU memory for efficient training. Adjust batch sizes if you run into memory issues.
- Learning Rate: Adjust learning rates depending on the performance observed in training. Lower learning rates may help stabilize training, especially in traditional fine-tuning.
- Data Preparation: Make sure the data is pre-processed properly, and any issues with tokenization (e.g., <|sep|> not found) are handled effectively.

## Model Weight Links(Saved to drive):-
- Soft Prompts:- https://drive.google.com/file/d/1V3brIBff3jme0JGhO5SRmXtCC2iJ3Ghv/view?usp=sharing
- LORA:- https://drive.google.com/file/d/1AWgaiMoIr69smSIQeVrDbnWV3FIkBgYE/view?usp=sharing
- Traditional Fine Tuning:- https://drive.google.com/file/d/1qrD34TNyZkHpAAjuka9kgdnJckzbpmeQ/view?usp=sharing

## Conclusion
This README provides all the essential steps to run, train, and evaluate the summarization models using three different fine-tuning methods. Follow the instructions carefully, and adjust configurations where necessary based on system resources and desired performance.
