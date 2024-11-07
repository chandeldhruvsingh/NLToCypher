import transformers
import peft
import torch
from datasets import load_dataset
import subprocess
import os

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"  
DATASET_NAME = "../data/training_data.json"  

# Create the output directory if it doesn't exist
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# Quantization
model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    ),
    trust_remote_code=True,
    device_map="cpu",
)

# Load tokenizer and set padding token
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# LoRA configuration
peft_config = peft.LoraConfig(
    task_type=peft.TaskType.CAUSAL_LM,
    inference_mode=False,
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=[
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "mlp.gate_proj", "mlp.down_proj", "mlp.up_proj"
    ]
)

# Apply LoRA to the model
model = peft.get_peft_model(model, peft_config)

# Load and preprocess the dataset
dataset = load_dataset("json", data_files=DATASET_NAME)

def preprocess_function(examples):
    inputs = [f"Question: {q}\nCypher:" for q in examples["Question"]]
    targets = [f" {c}" for c in examples["Cypher"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Split and tokenize the dataset
dataset = dataset["train"].train_test_split(test_size=0.1)
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = transformers.TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=2,
    optim="paged_adamw_32bit",
    save_steps=100,
    logging_steps=100,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)

# Trainer setup
trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Start training
trainer.train()

# Save the fine-tuned model and tokenizer
fine_tuned_model_path = os.path.join(output_dir, "fine_tuned_model")
model.save_pretrained(fine_tuned_model_path)
tokenizer.save_pretrained(fine_tuned_model_path)

# Merge LoRA weights with base model
# Check for a merge method; otherwise, save as is
merged_model = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
merged_model_path = os.path.join(output_dir, "merged_model")
merged_model.save_pretrained(merged_model_path)

print("fine tuning complete. converting to GGUF format...")

# Convert to GGUF format using llama.cpp
llama_cpp_path = "/Users/dhruvchandel/LLaMaCPP/llama.cpp/"  # Replace with your llama.cpp directory
gguf_path = os.path.join(output_dir, "NLToCypher.Q4.gguf")

# Convert to GGUF format
subprocess.run([
    f"{llama_cpp_path}/convert.py",
    "--outfile", gguf_path,
    "--outtype", "q4_0",
    merged_model_path
], check=True)

print(f"GGUF file saved at {gguf_path}")
