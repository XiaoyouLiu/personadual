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

def process_chunk(taskname, filename, chunk, pi):
    for i in chunk:
        if not i['answer']:
            i['extractor_answer'] = None
        elif taskname in ['AlignX', 'UltraMedical', 'flan']:
            matches = re.findall(r'\{[\s\S]*?\}', i['answer'])
            i['extractor_answer'] = matches[-1] if matches else None
        with open(f"./train-dataprocess/sft/temp/extractor_{filename}_temp{pi}.jsonl", 'a') as f:
            f.write(json.dumps(i, ensure_ascii=False)+'\n')
        print(i['extractor_answer'])
    return

if __name__ == '__main__':
    # par
    filename = 'flan_aligned'
    taskname = filename.split('_')[0]
    nnum = 20
    # load data
    with open(f'./train-dataprocess/sft/thinkingprocess/personalization_without_answer/{filename}.jsonl', 'r') as f:
            data = [json.loads(line) for line in f if line.strip()]
    # begin
    if os.path.exists(f"./train-dataprocess/sft/extractor/{filename}.jsonl") or os.path.exists(f"./train-dataprocess/sft/temp/extractor_{filename}_temp0.jsonl"):
        finished = []
        if os.path.exists(f"./train-dataprocess/sft/temp/extractor_{filename}_temp0.jsonl"):
            for i in range(nnum):
                try:
                    with open(f"./train-dataprocess/sft/temp/extractor_{filename}_temp{i}.jsonl", "r") as infile:
                        finished += [str(json.loads(line)['id']) for line in infile if line.strip()]
                except FileNotFoundError:
                    print(f"./train-dataprocess/sft/temp/extractor_{filename}_temp{i}.jsonl NOT FOUND")
        if os.path.exists(f"./train-dataprocess/sft/extractor/{filename}.jsonl"):
            with open(f"./train-dataprocess/sft/extractor/{filename}.jsonl", "r") as f:
                finished += [str(json.loads(line)['id']) for line in f if line.strip()]
    else:
        finished = []
    classify_data = [d for d in data if str(d['id']) not in finished]
    print(len(classify_data))
    chunk_size = len(classify_data) // nnum
    chunks = [classify_data[i:i + chunk_size] for i in range(0, len(classify_data), chunk_size)]
    with Pool(nnum) as pool:
        pool.starmap(process_chunk, [(taskname, filename, chunk, pi) for pi, chunk in enumerate(chunks)])
    with open(f"./train-dataprocess/sft/extractor/{filename}.jsonl", "a") as outfile:
        for i in range(nnum):
            with open(f"./train-dataprocess/sft/temp/extractor_{filename}_temp{i}.jsonl", "r") as infile:
                for line in infile:
                        line = line.strip()
                        if line:  # 跳过空行
                            outfile.write(line+ "\n")
            os.remove(f"./train-dataprocess/sft/temp/extractor_{filename}_temp{i}.jsonl")
    with open(f"./train-dataprocess/sft/extractor/{filename}.jsonl", 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]
    id_list = []
    with open(f"./train-dataprocess/sft/extractor/{filename}.jsonl", "w") as outfile:
        for i in data:
            if i['id'] not in id_list:
                outfile.write(json.dumps(i, ensure_ascii=False)+ "\n")
                id_list.append(i['id'])
        print(len(id_list))