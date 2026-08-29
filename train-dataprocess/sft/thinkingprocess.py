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

answer_format = {
    "AlignX": f"\"answer\":\"A/B\"",
    "UltraMedical": f"\"answer\":\"A/B/C/D\"",
    "flan_opt": f"\"answer\":\"A/B/...\"",
    "flan": f"\"answer\":\"your answer\"",
    "megascience": f"\"answer\":\"your answer\"",
    "openthoughts": f"\"answer\":\"your answer\""
}

prompt_format = {
    "general": Template(
        "Question: $question\n"
        "Groud-truth answer: $answer\n"
        "Your task is to generate a straight-line chain-of-thought (CoT) reasoning trace that:\n1.starts from the question;\n2.proceeds step-by-step without backtracking or restatements;\n3.Never contradicts the given ground-truth answer;\n4.arrives at the ground-truth answer.\n"
        "Please return your answer in JSON format:\n"
        """{"reasoning":"your thinking process"}"""
    ),
    "personalization": Template(
        "User persona: $persona\n"
        "User question: $question\n"
        "Groud-truth answer: $answer\n"
        "Your task is to generate a step-by-step chain-of-thought (CoT) reasoning trace that starts from the question and soundly arrives at the ground-truth answer. You need to clearly consider the user's persona in your reasoning trajectory and, if necessary, his preferences. But be sure to note that the user's persona is NOT your role and DON'T answer as the user's persona during the response process.\n"
        "Please return your answer in JSON format:\n"
        """{"reasoning":"your thinking process considering the user persona"}"""
    ),
    "personalization_without_answer": Template(
        "User persona: $persona\n"
        "User question: $question\n"
        "Please answer the user's question, and write down your thought process and answer. You need to clearly consider the user's persona in your thinking process and, if necessary, his preferences. But be sure to note that the user's persona is NOT your role and DON'T answer as the user's persona during the response process.\n"
        "Please return your answer in JSON format:\n"
        """{"thinking":"your thinking process considering the user persona", $answerformat}"""
    )
}

from ratelimit import limits, sleep_and_retry
import traceback
RPM_PER_PROCESS = 300/10

def generate_answer(prompt, api_key, base_url):
    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    # 调用模型
    try:
        chat_response = client.chat.completions.create(
            model = "qwen30b",
            messages=[{"role": "system", "content": prompt}],
            temperature = 0,
            max_tokens=4096*4
        )
        ans = chat_response.choices[0].message.content
    except Exception:
        print("发生错误:")
        traceback.print_exc()
    return ans

def process_chunk(mode, taskname, personatype, chunk, api_key, base_url, pi):
    for i in chunk:
        a = i['a']
        if mode == 'general':
            prompt = prompt_format[mode].substitute(question = i['q'], answer = a)
        elif mode == 'personalization_without_answer':
            if personatype == 'aligned':
                if taskname == 'flan' and 'OPTIONS' in i['q']:
                    prompt = prompt_format[mode].substitute(question = i['q'], persona = i['aligned_persona'], answerformat = answer_format['flan_opt'])
                else:
                    prompt = prompt_format[mode].substitute(question = i['q'], persona = i['aligned_persona'], answerformat = answer_format[taskname])
            elif personatype == 'unaligned':
                if taskname == 'flan' and 'OPTIONS' in i['q']:
                    prompt = prompt_format[mode].substitute(question = i['q'], persona = i['unaligned_persona'], answerformat = answer_format['flan_opt'])
                else:
                    prompt = prompt_format[mode].substitute(question = i['q'], persona = i['unaligned_persona'], answerformat = answer_format[taskname])
        elif mode == 'personalization':
            prompt = prompt_format[mode].substitute(question = i['q'], persona = i['aligned_persona'], answer = a)
        i['answer'] = generate_answer(prompt, api_key, base_url)
        with open(f"./train-dataprocess/temp/{mode}/{taskname}_{personatype}_temp{pi}.jsonl", 'a') as f:
            f.write(json.dumps(i, ensure_ascii=False)+'\n')
        print(i['answer'])
        # print(prompt)
    return

if __name__ == '__main__':
    api_key = "EMPTY"
    base_url = "http://localhost:8005/v1"
    # par
    mode = 'personalization_without_answer'
    # mode = 'general'
    taskname = 'openthoughts'
    personatype = 'aligned'
    nnum = 20
    # load data
    with open(f'./train-dataprocess/sft/data/{taskname}.json', 'r') as f:
            data = json.load(f)
    if mode == 'personalization':
        data = [d for d in data if mode in d and not d[mode]]
    elif mode == 'general':
        data = [d for d in data if not d[mode]]
    # begin
    if os.path.exists(f"./train-dataprocess/sft/thinkingprocess/{mode}/{taskname}_{personatype}.jsonl") or os.path.exists(f"./train-dataprocess/temp/{mode}/{taskname}_{personatype}_temp0.jsonl"):
        finished = []
        if os.path.exists(f"./train-dataprocess/sft/thinkingprocess/{mode}/{taskname}_{personatype}.jsonl"):
            with open(f"./train-dataprocess/sft/thinkingprocess/{mode}/{taskname}_{personatype}.jsonl", "r") as f:
                finished += [str(json.loads(line)['id']) for line in f if line.strip()]
        if os.path.exists(f"./train-dataprocess/temp/{mode}/{taskname}_{personatype}_temp0.jsonl"):
            for i in range(nnum):
                try:
                    with open(f"./train-dataprocess/temp/{mode}/{taskname}_{personatype}_temp{i}.jsonl", "r") as infile:
                        finished += [str(json.loads(line)['id']) for line in infile if line.strip()]
                except FileNotFoundError:
                    print(f"./train-dataprocess/temp/{mode}/{taskname}_{personatype}_temp{i}.jsonl NOT FOUND")
        print(len(finished))
    else:
        finished = []
    classify_data = [d for d in data if str(d['id']) not in finished]
    print(len(classify_data))
    chunk_size = len(classify_data) // nnum
    chunks = [classify_data[i:i + chunk_size] for i in range(0, len(classify_data), chunk_size)]
    with Pool(nnum) as pool:
        pool.starmap(process_chunk, [(mode, taskname, personatype, chunk, api_key, base_url, pi) for pi, chunk in enumerate(chunks)])
    with open(f"./train-dataprocess/sft/thinkingprocess/{mode}/{taskname}_{personatype}.jsonl", "a") as outfile:
        for i in range(nnum):
            with open(f"./train-dataprocess/temp/{mode}/{taskname}_{personatype}_temp{i}.jsonl", "r") as infile:
                for line in infile:
                        line = line.strip()
                        if line:  # 跳过空行
                            outfile.write(line+ "\n")
            os.remove(f"./train-dataprocess/temp/{mode}/{taskname}_{personatype}_temp{i}.jsonl")
    with open(f"./train-dataprocess/sft/thinkingprocess/{mode}/{taskname}_{personatype}.jsonl", 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]
    id_list = []
    with open(f"./train-dataprocess/sft/thinkingprocess/{mode}/{taskname}_{personatype}.jsonl", "w") as outfile:
        for i in data:
            if i['id'] not in id_list:
                outfile.write(json.dumps(i, ensure_ascii=False)+ "\n")
                id_list.append(i['id'])
        print(len(id_list))