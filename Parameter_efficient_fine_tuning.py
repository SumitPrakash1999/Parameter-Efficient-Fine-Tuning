#------------------------------------------------------------------------------------------------------------------------------------
#-----------SOFT PROMPTING CODE-----------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
import pandas as pd

train_data_path = '/kaggle/input/newspaper-text-summarization-cnn-dailymail/cnn_dailymail/train.csv'
validation_data_path = '/kaggle/input/newspaper-text-summarization-cnn-dailymail/cnn_dailymail/validation.csv'
test_data_path = '/kaggle/input/newspaper-text-summarization-cnn-dailymail/cnn_dailymail/test.csv'

train_data = pd.read_csv(train_data_path)
validation_data = pd.read_csv(validation_data_path)
test_data = pd.read_csv(test_data_path)

# Combining all datasets
full_data = pd.concat([train_data, validation_data, test_data], ignore_index=True)

# Reducing the combined dataset to 30,000 entries by sorting and selecting the shortest
def reduce_to_30000_shortest(df, text_column, summary_column):
    df['total_length'] = df[text_column].str.len() + df[summary_column].str.len()
    df_sorted = df.sort_values(by='total_length').reset_index(drop=True)
    reduced_df = df_sorted.head(30000)
    return reduced_df.drop(columns=['total_length'])

reduced_data = reduce_to_30000_shortest(full_data, 'article', 'highlights')

# Splitting into 21,000 for training, 6,000 for validation, and 3,000 for testing
train_data = reduced_data.iloc[:21000].reset_index(drop=True)
validation_data = reduced_data.iloc[21000:27000].reset_index(drop=True)
test_data = reduced_data.iloc[27000:].reset_index(drop=True)

print(f"Final Train Data Size: {len(train_data)} samples.")
print(f"Final Validation Data Size: {len(validation_data)} samples.")
print(f"Final Test Data Size: {len(test_data)} samples.")

# Converting datasets to appropriate format
class SummarizationDataset(Dataset):
    def __init__(self, articles, summaries, tokenizer, max_length=512):
        self.articles = articles
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        article = self.articles[idx]
        summary = self.summaries[idx]
        input_text = f"{article} <|sep|> {summary}"
        
#         # Debug print to check the input format
#         print(f"[INFO] Processing Sample {idx}: {input_text[:100]}...")
        
        inputs = self.tokenizer(
            input_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )

        sep_token_id = self.tokenizer.convert_tokens_to_ids('<|sep|>')
        if sep_token_id not in inputs.input_ids[0].tolist():
            print(f"[WARNING] <|sep|> not found in input for Sample {idx}. Skipping this sample.")
            return self.__getitem__((idx + 1) % len(self))  # Skip this sample and fetch the next one

        labels = inputs.input_ids.clone()
        labels[:, :inputs.input_ids[0].tolist().index(sep_token_id) + 1] = -100
        
        return {
            "input_ids": inputs.input_ids.flatten(),
            "attention_mask": inputs.attention_mask.flatten(),
            "labels": labels.flatten()
        }
# Initializing tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.add_special_tokens({'pad_token': '[PAD]', 'sep_token': '<|sep|>'})
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
model.resize_token_embeddings(len(tokenizer)) # To include new tokens
model.eval()
model.to("cuda") 

for param in model.parameters():
    param.requires_grad = False

# Converting datasets to PyTorch-compatible DataLoader
train_dataset = SummarizationDataset(train_data['article'], train_data['highlights'], tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

validation_dataset = SummarizationDataset(validation_data['article'], validation_data['highlights'], tokenizer)
validation_loader = DataLoader(validation_dataset, batch_size=8, shuffle=False)

test_dataset = SummarizationDataset(test_data['article'], test_data['highlights'], tokenizer)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

class SoftPromptEmbedding(torch.nn.Module):
    def __init__(self, num_prompts, embedding_dim, init_tokens, model, tokenizer):
        super(SoftPromptEmbedding, self).__init__()
        self.embeddings = torch.nn.Embedding(num_prompts, embedding_dim)
        input_embeddings = model.get_input_embeddings()
        for i, token in enumerate(init_tokens):
            token_id = tokenizer.encode(token, add_special_tokens=False)[0]
            self.embeddings.weight.data[i] = input_embeddings.weight.data[token_id]

    def forward(self, input_ids):
        return self.embeddings(input_ids)


soft_prompt = SoftPromptEmbedding(
    num_prompts=5, embedding_dim=model.config.hidden_size,
    init_tokens=["summarize", "article", "content", "news", "highlight"],
    model=model, tokenizer=tokenizer
).to("cuda")

optimizer = AdamW(soft_prompt.parameters(), lr=0.3)

import time

def train(model, tokenizer, soft_prompt, optimizer, train_loader, epochs=3):
    model.train()
    loss_fn = CrossEntropyLoss(ignore_index=-100) 
    
    total_training_time = 0 
    
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        print(f"Starting Epoch {epoch + 1}...")
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = {key: val.to("cuda") for key, val in batch.items()}
            
            print(f"[DEBUG] Processing Batch {batch_idx + 1}")

            prompt_ids = torch.arange(5).to("cuda") 
            prompt_embeds = soft_prompt(prompt_ids).unsqueeze(0).expand(inputs['input_ids'].size(0), -1, -1)
            
            inputs_embeds = torch.cat([prompt_embeds, model.get_input_embeddings()(inputs['input_ids'])], dim=1)
            
            prompt_attention_mask = torch.ones((inputs['input_ids'].size(0), prompt_embeds.size(1)), device="cuda")
            attention_mask = torch.cat([prompt_attention_mask, inputs['attention_mask']], dim=1)
            
            labels = inputs['labels']
            soft_prompt_padding = torch.full((labels.size(0), prompt_embeds.size(1)), -100, device="cuda")
            adjusted_labels = torch.cat([soft_prompt_padding, labels], dim=1)
            
            outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=adjusted_labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(soft_prompt.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            print(f"[DEBUG] Batch {batch_idx + 1} Loss: {loss.item()}")

        epoch_time = time.time() - start_time
        total_training_time += epoch_time
        print(f"Epoch {epoch + 1} Completed. Average Loss: {total_loss / len(train_loader)}. Time: {epoch_time:.2f} seconds")
    
    print(f"Total Training Time for {epochs} Epochs: {total_training_time:.2f} seconds.")

import pandas as pd
from tqdm import tqdm 


def evaluate(model, tokenizer, soft_prompt, data_loader, save_csv=False, csv_filename="evaluation_results.csv"):
    model.eval()
    total_loss = 0
    total_eval_time = 0  
    batch_dataframes = []  
    
    with torch.no_grad():
        start_time = time.time() 
        for batch_idx, batch in tqdm(enumerate(data_loader), desc="Evaluating Data"):
            inputs = {key: val.to("cuda") for key, val in batch.items()}

            prompt_ids = torch.arange(5).to("cuda")
            prompt_embeds = soft_prompt(prompt_ids).unsqueeze(0).expand(inputs['input_ids'].size(0), -1, -1)

            inputs_embeds = torch.cat([prompt_embeds, model.get_input_embeddings()(inputs['input_ids'])], dim=1)

            prompt_attention_mask = torch.ones((inputs['input_ids'].size(0), prompt_embeds.size(1)), device="cuda")
            attention_mask = torch.cat([prompt_attention_mask, inputs['attention_mask']], dim=1)

            labels = inputs['labels']
            soft_prompt_padding = torch.full((labels.size(0), prompt_embeds.size(1)), -100, device="cuda")
            adjusted_labels = torch.cat([soft_prompt_padding, labels], dim=1)

            outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=adjusted_labels)
            loss = outputs.loss
            total_loss += loss.item()

            if save_csv:
                preds = torch.argmax(outputs.logits, dim=-1)
                decoded_preds = tokenizer.batch_decode(preds.cpu(), skip_special_tokens=True)
                
                labels = inputs['labels'].detach().cpu().numpy()
                decoded_labels = []
                for label in labels:
                    valid_tokens = label[label != -100]
                    decoded_string = tokenizer.decode(valid_tokens, skip_special_tokens=True)
                    decoded_labels.append(decoded_string)

                batch_df = pd.DataFrame({
                    'Input': tokenizer.batch_decode(inputs['input_ids'].cpu(), skip_special_tokens=True),
                    'Predicted Summary': decoded_preds,
                    'Actual Summary': decoded_labels
                })

                batch_dataframes.append(batch_df)

        total_eval_time = time.time() - start_time
        print(f"Total Evaluation Time: {total_eval_time:.2f} seconds")

    if save_csv:
        final_output_df = pd.concat(batch_dataframes, ignore_index=True)
        final_output_df.to_csv(csv_filename, index=False)
        print(f"Evaluation results saved to '{csv_filename}'")
    
    avg_loss = total_loss / len(data_loader)
    print(f"Average Loss: {avg_loss}")
    return avg_loss


train(model, tokenizer, soft_prompt, optimizer, train_loader, epochs=6)

print("Evaluating on Validation Set:")
evaluate(model, tokenizer, soft_prompt, validation_loader)

print("Evaluating on Test Set and Saving to CSV:")
test_results = evaluate(model, tokenizer, soft_prompt, test_loader, save_csv=True, csv_filename="soft_prompt_test_results.csv")

def save_soft_prompt(soft_prompt, path="soft_prompt.pt"):
    torch.save(soft_prompt.state_dict(), path)
    print(f"Soft prompt embeddings saved to {path}")

save_soft_prompt(soft_prompt, "soft_prompt_trained.pt")

def load_soft_prompt(soft_prompt, path="soft_prompt_trained.pt"):
    soft_prompt.load_state_dict(torch.load(path))
    soft_prompt.to("cuda") 
    print(f"Soft prompt embeddings loaded from {path}")

load_soft_prompt(soft_prompt, "soft_prompt_trained.pt")

# pip install rouge-score
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


rouge_scores = calculate_rouge_from_csv("soft_prompt_test_results.csv")







    
    
#------------------------------------------------------------------------------------------------------------------------------------
#-----------LORA CODE----------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------

# !pip install peft
import time
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig
import pandas as pd

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.add_special_tokens({'pad_token': '[PAD]', 'sep_token': '<|sep|>'})
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
model.resize_token_embeddings(len(tokenizer))
model.eval()
model.to("cuda")


for param in model.parameters():
    param.requires_grad = False

# Configuring LoRA using PEFT library
lora_config = LoraConfig(
    r=4,  # Rank of the low-rank matrix
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.1, 
    target_modules=["mlp.c_fc", "mlp.c_proj"], 
    bias="none"
)

model = get_peft_model(model, lora_config)

class SummarizationDataset(torch.utils.data.Dataset):
    def __init__(self, articles, summaries, tokenizer, max_length=512):
        self.articles = articles
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.articles)
    
    def __getitem__(self, idx):
        input_text = self.articles[idx] + " <|sep|> " + self.summaries[idx]
        inputs = self.tokenizer(
            input_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )
        labels = inputs.input_ids.clone()
        
        sep_token_id = self.tokenizer.convert_tokens_to_ids('<|sep|>')
        try:
            sep_index = inputs.input_ids[0].tolist().index(sep_token_id) + 1
            labels[:, :sep_index] = -100
        except ValueError:
            print(f"[WARNING] <|sep|> token not found in input: {input_text[:50]}... Skipping sample.")
            labels[:] = -100
        
        return {
            "input_ids": inputs.input_ids.flatten(),
            "attention_mask": inputs.attention_mask.flatten(),
            "labels": labels.flatten()
        }


train_dataset = SummarizationDataset(train_data['article'], train_data['highlights'], tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

validation_dataset = SummarizationDataset(validation_data['article'], validation_data['highlights'], tokenizer)
validation_loader = DataLoader(validation_dataset, batch_size=8, shuffle=False)


optimizer = AdamW(model.parameters(), lr=0.001)

def train_lora(model, train_loader, epochs=3):
    model.train()
    loss_fn = CrossEntropyLoss(ignore_index=-100)  
    total_training_time = 0  
    
    for epoch in range(epochs):
        start_time = time.time()  
        total_loss = 0
        print(f"Starting Epoch {epoch + 1}...")
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = {key: val.to("cuda") for key, val in batch.items()}
            
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"]
            )
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx + 1}, Loss: {loss.item()}")

        epoch_time = time.time() - start_time
        total_training_time += epoch_time
        print(f"Epoch {epoch + 1} Completed. Average Loss: {total_loss / len(train_loader)}. Time: {epoch_time:.2f} seconds")
    
    print(f"Total Training Time for {epochs} Epochs: {total_training_time:.2f} seconds.")

import pandas as pd
from tqdm import tqdm 

def evaluate_lora(model, tokenizer, data_loader, save_csv=False, csv_filename="lora_evaluation_results.csv"):
    model.eval()
    total_loss = 0
    total_eval_time = 0 
    batch_dataframes = []  
    
    with torch.no_grad():
        start_time = time.time() 
        for batch_idx, batch in tqdm(enumerate(data_loader), desc="Evaluating Data"):
            inputs = {key: val.to("cuda") for key, val in batch.items()}

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"]
            )
            loss = outputs.loss
            total_loss += loss.item()

            if save_csv:
                preds = torch.argmax(outputs.logits, dim=-1)

                decoded_preds = tokenizer.batch_decode(preds.cpu(), skip_special_tokens=True)

                labels = inputs['labels'].detach().cpu().numpy()
                decoded_labels = []
                for label in labels:
                    valid_tokens = label[(label != -100)]
                    decoded_string = tokenizer.decode(valid_tokens, skip_special_tokens=True)
                    decoded_labels.append(decoded_string)

                
                batch_df = pd.DataFrame({
                    'Input': tokenizer.batch_decode(inputs['input_ids'].cpu(), skip_special_tokens=True),
                    'Predicted Summary': decoded_preds,
                    'Actual Summary': decoded_labels
                })

                batch_dataframes.append(batch_df)

        total_eval_time = time.time() - start_time 
        print(f"Total Evaluation Time: {total_eval_time:.2f} seconds")

    if save_csv:
        final_output_df = pd.concat(batch_dataframes, ignore_index=True)
        final_output_df.to_csv(csv_filename, index=False)
        print(f"Evaluation results saved to '{csv_filename}'")
    
    avg_loss = total_loss / len(data_loader)
    print(f"Average Loss: {avg_loss}")
    return avg_loss

print("Training with LoRA...")
train_lora(model, train_loader, epochs=6)

print("Evaluating on Validation Set:")
evaluate_lora(model, tokenizer, validation_loader)

print("Evaluating on Test Set and Saving to CSV:")
evaluate_lora(model, tokenizer, test_loader, save_csv=True, csv_filename="lora_test_results.csv")

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

rouge_scores = calculate_rouge_from_csv("lora_test_results.csv")

# Function to save the model and LoRA components
def save_lora_model(model, save_path):
    # Save the LoRA components
    lora_save_path = save_path + "_lora.pt"
    torch.save(model.state_dict(), lora_save_path)
    print(f"LoRA components saved to {lora_save_path}")

    # Save the full model
    full_model_save_path = save_path + "_full_model.pt"
    model.save_pretrained(full_model_save_path)
    print(f"Full model state saved to {full_model_save_path}")

# # Function to load the LoRA components
# def load_lora_model(model, load_path):
#     # Load the LoRA components
#     lora_load_path = load_path + "_lora.pt"
#     model.load_state_dict(torch.load(lora_load_path))
#     print(f"LoRA components loaded from {lora_load_path}")

save_path = "./lora_gpt_neo"
save_lora_model(model, save_path)

# To reload:
# model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
# model = get_peft_model(model, lora_config)
# load_lora_model(model, save_path)




#------------------------------------------------------------------------------------------------------------------------------------
#-----------TRADITIONAL FINE TUNING CODE---------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------

import time
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.add_special_tokens({'pad_token': '[PAD]', 'sep_token': '<|sep|>'})
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
model.resize_token_embeddings(len(tokenizer))  # To include new tokens
model.to("cuda")

# Freezing all parameters in the model
for param in model.parameters():
    param.requires_grad = False

# Unfreezing the language modeling head (LM head)
for param in model.lm_head.parameters():
    param.requires_grad = True

class SummarizationDataset(torch.utils.data.Dataset):
    def __init__(self, articles, summaries, tokenizer, max_length=512):
        self.articles = articles
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.articles)
    
    def __getitem__(self, idx):
        input_text = self.articles[idx] + " <|sep|> " + self.summaries[idx]
        inputs = self.tokenizer(
            input_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )
        labels = inputs.input_ids.clone()
        sep_token_id = self.tokenizer.convert_tokens_to_ids('<|sep|>')
        try:
            sep_index = inputs.input_ids[0].tolist().index(sep_token_id) + 1
            labels[:, :sep_index] = -100
        except ValueError:
            print(f"[WARNING] <|sep|> token not found in input: {input_text[:50]}... Skipping sample.")
            labels[:] = -100

        return {
            "input_ids": inputs.input_ids.flatten(),
            "attention_mask": inputs.attention_mask.flatten(),
            "labels": labels.flatten()
        }

train_dataset = SummarizationDataset(train_data['article'], train_data['highlights'], tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

validation_dataset = SummarizationDataset(validation_data['article'], validation_data['highlights'], tokenizer)
validation_loader = DataLoader(validation_dataset, batch_size=8, shuffle=False)

optimizer = AdamW(model.lm_head.parameters(), lr=0.001)

def train_finetune(model, train_loader, epochs=3):
    model.train()
    loss_fn = CrossEntropyLoss(ignore_index=-100)  
    total_training_time = 0  
    
    for epoch in range(epochs):
        start_time = time.time() 
        total_loss = 0
        print(f"Starting Epoch {epoch + 1}...")
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = {key: val.to("cuda") for key, val in batch.items()}
            
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"]
            )
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.lm_head.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx + 1}, Loss: {loss.item()}")

        epoch_time = time.time() - start_time
        total_training_time += epoch_time
        print(f"Epoch {epoch + 1} Completed. Average Loss: {total_loss / len(train_loader)}. Time: {epoch_time:.2f} seconds")
    
    print(f"Total Training Time for {epochs} Epochs: {total_training_time:.2f} seconds.")


print("Training with Traditional Fine-Tuning...")
train_finetune(model, train_loader, epochs=10)

import pandas as pd
from tqdm import tqdm 

# Updated Evaluation Function
def evaluate_finetune(model, tokenizer, data_loader, save_csv=False, csv_filename="finetune_test_results.csv"):
    model.eval()
    batch_dataframes = []
    total_loss = 0
    total_eval_time = 0 

    with torch.no_grad():
        start_time = time.time()
        for batch_idx, batch in tqdm(enumerate(data_loader), desc="Evaluating Data"):
            inputs = {key: val.to("cuda") for key, val in batch.items()}

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"]
            )
            loss = outputs.loss
            total_loss += loss.item()

            if save_csv:
                
                preds = torch.argmax(outputs.logits, dim=-1)

                decoded_preds = tokenizer.batch_decode(preds.cpu(), skip_special_tokens=True)

                labels = inputs['labels'].detach().cpu().numpy()
                decoded_labels = []
                for label in labels:
                    valid_tokens = label[(label != -100)]
                    decoded_string = tokenizer.decode(valid_tokens, skip_special_tokens=True)
                    decoded_labels.append(decoded_string)

                batch_df = pd.DataFrame({
                    'Input': tokenizer.batch_decode(inputs['input_ids'].cpu(), skip_special_tokens=True),
                    'Predicted Summary': decoded_preds,
                    'Actual Summary': decoded_labels
                })

                batch_dataframes.append(batch_df)

        total_eval_time = time.time() - start_time 
        print(f"Total Evaluation Time: {total_eval_time:.2f} seconds")

    
    if save_csv:
        final_output_df = pd.concat(batch_dataframes, ignore_index=True)
        final_output_df.to_csv(csv_filename, index=False)
        print(f"Evaluation results saved to '{csv_filename}'")
        return final_output_df

    avg_loss = total_loss / len(data_loader)
    print(f"Average Loss: {avg_loss}")
    return avg_loss

# Evaluate on Validation Set
print("Evaluating on Validation Set:")
evaluate_finetune(model, tokenizer, validation_loader, save_csv=True, csv_filename="finetune_validation_results.csv")

# Evaluate on Test Set
print("Evaluating on Test Set and Saving to CSV:")
evaluate_finetune(model, tokenizer, test_loader, save_csv=True, csv_filename="finetune_test_results.csv")

from rouge_score import rouge_scorer
import pandas as pd

# Function to calculate ROUGE scores from saved CSV file
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

rouge_scores = calculate_rouge_from_csv("finetune_test_results.csv")

# Function to save fine-tuned model weights
def save_finetuned_model(model, path="finetuned_model.pt"):
    torch.save(model.state_dict(), path)
    print(f"Fine-tuned model weights saved to {path}")


save_finetuned_model(model, "finetuned_model_traditional.pt")

# # Function to load fine-tuned model weights
# def load_finetuned_model(model, path="finetuned_model_traditional.pt"):
#     model.load_state_dict(torch.load(path))
#     model.to("cuda")  # Ensure the model is moved to the correct device
#     print(f"Fine-tuned model weights loaded from {path}")


# load_finetuned_model(model, "finetuned_model_traditional.pt")



import torch

# Function to get trainable parameters count
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to monitor GPU memory usage
def get_gpu_memory_usage():
   
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # In GB
        memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  
        return memory_allocated, memory_reserved
    else:
        return 0, 0

# Function to simulate a forward pass for GPU usage measurement
def simulate_forward_pass(model, data_loader, num_batches=5):
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()  
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_batches:  
                break
            inputs = {key: val.to("cuda") for key, val in batch.items()}
            _ = model(**inputs)
        memory_allocated, memory_reserved = get_gpu_memory_usage()
        print(f"GPU Memory Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")


# 1. Soft Prompt Tuning
print("### Soft Prompt Tuning ###")
print(f"Trainable Parameters (Soft Prompt): {count_trainable_parameters(soft_prompt)}")
simulate_forward_pass(model, test_loader)  # Test loader used to measure during inference

# # 2. LoRA Tuning
# print("### LoRA Tuning ###")
# print(f"Trainable Parameters (LoRA): {count_trainable_parameters(model)}")
# simulate_forward_pass(model, test_loader)

# # 3. Traditional Fine-Tuning
# print("### Traditional Fine-Tuning ###")
# print(f"Trainable Parameters (Traditional Fine-Tuning): {count_trainable_parameters(model)}")
# simulate_forward_pass(model, test_loader)


