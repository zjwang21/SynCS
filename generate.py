from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

path = "Qwen2.5-3B-Instruct-zhrobn"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)

model.eval().cuda()

prompt_dic = {
    "annotation-zh2en": "Given a Chinese sentence, generate an English-annotated Chinese sentence. Annotation is the use of words from another language to explain certain words in a sentence. \n[Chinese Sentence]: {}",
    "annotation-en2zh": "Given an English sentence, generate an Chinese-annotated English sentence. Annotation is the use of words from another language to explain certain words in a sentence. \n[English Sentence]: {}",
    "replace-zh2en": "Given a Chinese sentence, generate a Chinese and English code-switching sentence. Code-switching is the use of more than one linguistic variety in a manner consistent with the syntax and phonology of each variety.\n[Chinese Sentence]: {}",
    "replace-en2zh": "Given an English sentence, generate a Chinese and English code-switching sentence. Code-switching is the use of more than one linguistic variety in a manner consistent with the syntax and phonology of each variety.\n[English Sentence]: {}",
    "annotation-ro2en": "Given a Romanian sentence, generate an English-annotated Romanian sentence. Annotation is the use of words from another language to explain certain words in a sentence. \n[Romanian Sentence]: {}",
    "annotation-en2ro": "Given an English sentence, generate a Romanian-annotated English sentence. Annotation is the use of words from another language to explain certain words in a sentence. \n[English Sentence]: {}",
    "replace-ro2en": "Given a Romanian sentence, generate a Romanian and English code-switching sentence. Code-switching is the use of more than one linguistic variety in a manner consistent with the syntax and phonology of each variety.\n[Romanian Sentence]: {}",
    "replace-en2ro": "Given an English sentence, generate an English and Romanian code-switching sentence. Code-switching is the use of more than one linguistic variety in a manner consistent with the syntax and phonology of each variety.\n[English Sentence]: {}",
    "annotation-bn2en": "Given a Bengali sentence, generate an English-annotated Bengali sentence. Annotation is the use of words from another language to explain certain words in a sentence. \n[Bengali Sentence]: {}",
    "annotation-en2bn": "Given an English sentence, generate a Bengali-annotated English sentence. Annotation is the use of words from another language to explain certain words in a sentence. \n[English Sentence]: {}",
    "replace-bn2en": "Given a Bengali sentence, generate a Bengali and English code-switching sentence. Code-switching is the use of more than one linguistic variety in a manner consistent with the syntax and phonology of each variety.\n[Bengali Sentence]: {}",
    "replace-en2bn": "Given an English sentence, generate an English and Bengali code-switching sentence. Code-switching is the use of more than one linguistic variety in a manner consistent with the syntax and phonology of each variety.\n[English Sentence]: {}",
}

while True:
    # 获取用户输入s
    user_input = input("你: ")
    
    if user_input.lower() in ["exit", "quit", "退出"]:
        print("聊天结束。再见！")
        break

    # 添加用户输入到会话中
    #query = prompt + prompt_affix.format(user_input)
    query = prompt_dic['replace-en2bn'].format(user_input) #template.format_map({"user_input": user_input})
    print(query)
    #messages = [
    #    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    #    {"role": "user", "content": query}
    #]
    #text = tokenizer.apply_chat_template(
    #    messages,
    #    tokenize=False,
    #    add_generation_prompt=True
    #)
    text = query
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        generation_config=GenerationConfig(
            do_sample=False,
            max_new_tokens=2048,
        )
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    print(generated_ids)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"assitant: {response}")