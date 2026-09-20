import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_PATH = r"C:\models-Qwen-del\Qwen3-0.6B"
LORA_PATH = r"C:\models-Qwen-del\output"   # 训练输出目录

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", trust_remote_code=True, local_files_only=True,
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

def generate_answer(question):
    messages = [
        {"role": "system", "content": "你是一个网络安全法专家，回答要准确、简洁。"},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    # 只取新生成的部分
    gen = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

# 测试
q = "企业未履行网络安全保护义务会有什么法律责任？"
print("问题:", q)
print("回答:", generate_answer(q))