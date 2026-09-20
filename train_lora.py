import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# ============ 1. 路径配置 ============
MODEL_PATH = r"C:\models-Qwen-del\Qwen3-0.6B"
DATA_PATH = r"C:\models-Qwen-del\data\train.jsonl"
OUTPUT_DIR = r"C:\models-Qwen-del\output"

# ============ 2. 加载数据 ============
def load_jsonl_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if not (isinstance(item, dict) and "messages" in item):
                    print(f"第 {line_num} 行无效: 缺少 messages 字段")
                    continue
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行 JSON 解析失败: {e}")
    if not data:
        raise ValueError("未加载到有效数据")
    print(f"成功加载 {len(data)} 条有效数据")
    return data

# ============ 3. 加载 tokenizer 和模型（4-bit QLoRA）============
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, trust_remote_code=True, local_files_only=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
)

# ============ 4. 应用 Qwen3 官方对话模板 ============
def format_chat(item):
    text = tokenizer.apply_chat_template(
        item["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

raw_data = load_jsonl_data(DATA_PATH)
formatted_data = [format_chat(item) for item in raw_data]

print("\n== 格式化后的数据示例 ==")
print(formatted_data[0]["text"][:500])

# ============ 5. 配置 LoRA ============
lora_config = LoraConfig(
    r=8,                                # 0.6B 模型 r=8 够用，显存不够可降到 4
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============ 6. 构建数据集 ============
dataset = Dataset.from_list(formatted_data)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        max_length=1024,          # 8GB 显存建议 512，长文本可升到 1024
        truncation=True,
    )

tokenized_dataset = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)
print("数据集分词完成")

# ============ 7. 训练参数 ============
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,       # 8GB 显存 + 4bit 建议 1
    gradient_accumulation_steps=8,       # 等效 batch = 8
    learning_rate=2e-4,
    num_train_epochs=8,                  # 小数据集多轮
    logging_steps=5,
    save_steps=20,
    save_total_limit=2,
    fp16=False,                          # 4bit 下用 bf16，Windows 建议关 fp16
    bf16=True,
    optim="paged_adamw_8bit",            # 8bit 优化器省显存
    report_to="none",
    overwrite_output_dir=True,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# ============ 8. 启动训练 ============
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()

# ============ 9. 保存 LoRA 权重 ============
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"训练完成，LoRA 权重已保存至: {OUTPUT_DIR}")