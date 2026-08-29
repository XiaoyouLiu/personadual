from openai import OpenAI
import json
import os
import json
import pandas as pd
from datetime import datetime
import re
import warnings
import numpy as np
from multiprocessing import Pool
import random
from string import Template
import traceback


prompt_format = {
    "megascience": Template(
        "You are an answer-validation expert, skilled at judging whether a candidate answer matches the ground-truth answer for a given question.\n"
        "Your will receive a user question, the ground-truth answer, and a candidate answer. Your task is to compare the candidate answer to the ground-truth answer and classify its correctness into exactly one of the two labels: correct, incorrect.\n\n"
        "USER_QUESTION: $question\n"
        "Ground-truth ANSWER: $answer\n"
        "Candidate ANSWER: $candidate\n\n"
        "Please return your judgement in JSON format:\n"
        """{"judgement":<correct/incorrect>}"""
    )
}


def generate_answer(prompt, api_key, base_url):
    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    # 调用模型
    try:
        chat_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            temperature=0
        )
    except Exception:
        print("发生错误:")
        traceback.print_exc()
    # 打印响应
    return chat_response.choices[0].message.content

def process_chunk(taskname, filename, chunk, api_key, base_url, pi):
    for i in chunk:
        if not i['answer']:
            i['extractor_answer'] = None
        else:
            if '</think>' in i['answer']:
                text = i['answer'].split('</think>')[-1].strip('\n\n')
            else:
                matches = re.findall(r'\{[\s\S]*?\}', i['answer'])
                if matches:
                    text = matches[-1]
                else:
                    text = i['answer']
            try:
                text = json.loads(text).get('answer')
            except json.decoder.JSONDecodeError:
                text = text
            prompt = prompt_format['megascience'].substitute(question = i['q'], answer = i['a'], candidate = text)
            # print(prompt)
            i['extractor_answer'] = generate_answer(prompt, api_key, base_url)
        with open(f"./train-dataprocess/temp/extractor_{filename}_temp{pi}.jsonl", 'a') as f:
            f.write(json.dumps(i, ensure_ascii=False)+'\n')
        print(i['extractor_answer'])
    return

if __name__ == '__main__':
    api_key = ""
    base_url = ""
    # par
    taskname = 'megascience'
    filename = 'megascience_aligned'
    nnum = 20
    # load data
    with open(f'./train-dataprocess/sft/thinkingprocess/personalization_without_answer/{taskname}_aligned.jsonl', 'r') as f:
            data = [json.loads(line) for line in f if line.strip()]
    # begin
    if os.path.exists(f"./train-dataprocess/sft/extractor/{filename}.jsonl") or os.path.exists(f"./train-dataprocess/temp/extractor_{filename}_temp0.jsonl"):
        finished = []
        if os.path.exists(f"./train-dataprocess/temp/extractor_{filename}_temp0.jsonl"):
            for i in range(nnum):
                try:
                    with open(f"./train-dataprocess/temp/extractor_{filename}_temp{i}.jsonl", "r") as infile:
                        finished += [str(json.loads(line)['id']) for line in infile if line.strip()]
                except FileNotFoundError:
                    print(f"./train-dataprocess/temp/extractor_{filename}_temp{i}.jsonl NOT FOUND")
        if os.path.exists(f"./train-dataprocess/sft/extractor/{filename}.jsonl"):
            with open(f"./train-dataprocess/sft/extractor/{filename}.jsonl", "r") as f:
                finished += [str(json.loads(line)['id']) for line in f if line.strip()]
    else:
        finished = []
    # classify_data = [d for d in data if str(d['id']) not in finished]
    # print(len(classify_data))
    # chunk_size = len(classify_data) // nnum
    # chunks = [classify_data[i:i + chunk_size] for i in range(0, len(classify_data), chunk_size)]
    # with Pool(nnum) as pool:
    #     pool.starmap(process_chunk, [(taskname, filename, chunk, api_key, base_url, pi) for pi, chunk in enumerate(chunks)])
    with open(f"./train-dataprocess/sft/extractor/{filename}.jsonl", "a") as outfile:
        for i in range(nnum):
            with open(f"./train-dataprocess/temp/extractor_{filename}_temp{i}.jsonl", "r") as infile:
                for line in infile:
                        line = line.strip()
                        if line:  # 跳过空行
                            outfile.write(line+ "\n")
            os.remove(f"./train-dataprocess/temp/extractor_{filename}_temp{i}.jsonl")
    with open(f"./train-dataprocess/sft/extractor/{filename}.jsonl", 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]
    id_list = []
    with open(f"./train-dataprocess/sft/extractor/{filename}.jsonl", "w") as outfile:
        for i in data:
            if i['id'] not in id_list:
                outfile.write(json.dumps(i, ensure_ascii=False)+ "\n")
                id_list.append(i['id'])
        print(len(id_list))