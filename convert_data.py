import json

INPUT_FILE  = r"C:\models-Qwen-del\data\温惜璇-32406400051-SFT数据集.txt"
OUTPUT_FILE = r"C:\models-Qwen-del\data\train.jsonl"

# 固定的 system 提示词（取自数据里 instruction 的角色设定）
SYSTEM_PROMPT = "你是一位中国网络安全领域的法律专家。"

count = 0
with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line_num, line in enumerate(fin, 1):
        line = line.strip()
        if not line:
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"第 {line_num} 行解析失败: {e}")
            continue

        instruction = item.get("instruction", "").strip()
        input_text  = item.get("input", "").strip()
        output_text = item.get("output", "").strip()

        if not output_text:
            print(f"第 {line_num} 行缺少 output，跳过")
            continue

        # 把 instruction 和 input 拼成 user 消息
        user_content = instruction
        if input_text:
            user_content += "\n\n【案例事实】\n" + input_text

        chatml = {
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": output_text},
            ]
        }

        fout.write(json.dumps(chatml, ensure_ascii=False) + "\n")
        count += 1
print(f"转换完成，共写入 {count} 条数据到 {OUTPUT_FILE}")