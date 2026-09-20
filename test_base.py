import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = r"C:\models-Qwen-del\Qwen3-0.6B"

# 1. 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, trust_remote_code=True, local_files_only=True
)

# 2. 加载模型（0.6B 很小，直接 bf16 全精度加载到 GPU 即可）
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
)
model.eval()

print("模型加载成功！设备:", next(model.parameters()).device)

# 3. 定义推理函数
def chat(question, system="你是一个有用的AI助手。"):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    # 用 Qwen3 官方对话模板构造 prompt
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    # 只取新生成的部分（去掉输入 prompt）
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

# 4. 测试几个问题
if __name__ == "__main__":
    questions = [
        "你好，请介绍一下你自己。",
        "你是一位中国网络安全领域的法律专家。请依据《中华人民共和国网络安全法》（2016年发布版及2025年修订版）及相关法律法规，对以下人民法院案例库入库案例进行全面分析:2019年9月至2022年5月间，杨某鹏利用其注册的某固传媒公司、某意科技公司、某固科技公司等公司研发的平台，招募数量庞大的兼职人员充当“网络水军”，并通过组织、操纵“网络水军”“养号”等方式，开展有偿“转评赞”“直发”“投诉举报”等业务，即在未核实信息真实性的情况下，实施包括对客户指定的影视作品、网络视频、游戏作品、商品的宣发等正面点赞、转发、评论，按客户要求在相关网络平台发布关于特定作品、商品的具体内容等提升热度的业务，以及通过在信息发布平台进行投诉举报等方式删帖以降低针对特定作品、商品的负面信息热度的业务。经调查，杨某鹏等共“养号”1294个，完成“转评赞”“直发”任务24万余条，任务金额合计896万余元；完成“投诉举报”任务1200余条，任务金额合计19万余元。另，杨某鹏、某固传媒公司等以营利为目的，有偿提供删帖服务的行为已被生效刑事判决认定构成犯罪。杭州市滨江区人民检察院在履行公益诉讼公告程序后依法向人民法院提起民事公益诉讼，请求法院判令：杨某鹏、某固传媒公司、某意科技公司、某固科技公司等四被告共同承担公益损害赔偿金100万元，删除已发布的虚假信息并注销相关网络账号，以及在国家级新闻媒体上公开赔礼道歉。。\n请按以下七个步骤依次作答：\n第一步：指明案例中的违法责任主体；\n第二步：依据《中华人民共和国网络安全法》（2025年修订版），分析违法责任主体的违法情况，需指明所引用的具体法律条款编号和对应内容；\n第三步：先依据《中华人民共和国网络安全法》（2025年修订版）说明对违法责任主体的处罚依据，再依据《中华人民共和国网络安全法》（2016年发布版）说明处罚依据；\n第四步：对比第三步的两种处罚依据，说明其差异；\n第五步：如果依据《中华人民共和国网络安全法》（2025年修订版），这个案例中的违法责任主体如何处罚合理？\n第六步：依据《中华人民共和国网络安全法》（2025年修订版），这个案例的处罚是否完备？请从各法条之间的逻辑关系出发，逐一分析是否存在遗漏的处罚措施；\n第七步：针对本案例，延伸提出两个值得思考的问题，并给出答案。"
    ]
    for q in questions:
        print("=" * 60)
        print("问题:", q)
        print("-" * 60)
        print("回答:", chat(q))
        print()