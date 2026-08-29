import re
import os
import pandas as pd
import numpy as np
import json
import tqdm
import yaml
import random
from pathlib import Path
from collections import defaultdict

def extract_outermost_json(text: str):
    """
    从 text 中提取最外层 JSON 对象/数组。
    返回 (python-dict/list, 原始字符串) 或 (None, None)
    """
    # 1. 找到第一个左大括号/中括号
    start = text.find('{')
    if start == -1:
        return None, None

    # 2. 逐字符扫描，计数括号深度
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:               # 最外层闭合
                json_str = text[start:i+1]
                try:
                    return json.loads(json_str), json_str
                except ValueError:
                    return None, None
    return None, None

'''
filename = 'megascience'
mode = 'general'
with open(f'./train-dataprocess/sft/thinkingprocess/{mode}/{filename}_aligned.jsonl', 'r') as f:
    data = [json.loads(line) for line in f if line.strip()]
data_id = [str(d['id']) for d in data]
id_reason = {}
for index in range(len(data)):
    try:
        py_obj, _ = extract_outermost_json(data[index]['answer'].split('</think>')[-1])
        reason = py_obj.get('reasoning') if py_obj else None
        if not reason and "\"reasoning\":" in data[index]['answer'].split('</think>')[-1]:
            try:
                reason = re.search(r'(?s)"reasoning"\s*:\s*"((?:\\.|[^"\\])*)"\s*}', data[index]['answer'].split('</think>')[-1])
                raw = reason.group(1)  
                reason = re.sub(r'\\(.)', r'\1', raw)
            except AttributeError:
                pass
        id_reason[str(data[index]['id'])] = reason
    except json.decoder.JSONDecodeError:
        matches = re.findall(r'\{[\s\S]*?\}', data[index]['answer'])
        print(matches[-1])
with open(f'./train-dataprocess/sft/data/{filename}.json', 'r') as f:
        total = json.load(f)
for index in range(len(total)):
    # if str(total[index]['id']) in data_id:
    if str(total[index]['id']) in data_id and not total[index][mode]:
        total[index][mode] = id_reason[str(total[index]['id'])]
print([d['id'] for d in total if str(d['id']) in data_id and not d[mode]])
print(len([d['id'] for d in total if str(d['id']) in data_id and not d[mode]]))
with open(f'./train-dataprocess/sft/data/{filename}.json', 'w') as f:
    f.write(json.dumps(total, ensure_ascii=False, indent=4))
'''
############# 采样划分为个性化模式的通用数据
'''
with open('./train-dataprocess/sft/data/flan2.json', 'r') as f:
    data = json.load(f)
task_counts = defaultdict(int)
task_total = defaultdict(int)
# 遍历每条记录
for item in data:
    task = item.get("task")
    if task is None:
        continue  # 跳过没有 task 的项
    task_total[task] += 1
    if "personalization" in item:
        task_counts[task] += 1
# 合并结果（包括没有 personalization 的 task）
result = {task: task_counts.get(task, 0) for task in task_total}
# 打印结果
print(json.dumps(result, indent=2, ensure_ascii=False))
'''
'''
INPUT_FILE  = './train-dataprocess/sft/data/flan2.json'     # 原始数据
OUTPUT_FILE = './train-dataprocess/sft/data/flan2.json'   # 采样结果
SEED        = 42                   # 可复现
TOTAL_NEED  = 2000
random.seed(SEED)
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)
filtered = [rec for rec in data
            if 'personalization' in rec and 'task' in rec]
groups = defaultdict(list)
for rec in filtered:
    groups[rec['task']].append(rec)
task_list = sorted(groups)
print('Task counts (with personalization):', {t: len(groups[t]) for t in task_list})
total_avail = sum(len(groups[t]) for t in task_list)
quotas = {t: round(TOTAL_NEED * len(groups[t]) / total_avail)
              for t in task_list}
    # 四舍五入后可能不是正好 1000，手动把差值调到最大组
delta = TOTAL_NEED - sum(quotas.values())
max_task = max(quotas, key=lambda t: quotas[t])
quotas[max_task] += delta
print('Quotas:', quotas)
sample = []
for t in task_list:
    n_need = quotas[t]
    pool = groups[t]
    if len(pool) < n_need:
        raise ValueError(f'Task {t} only has {len(pool)} records with personalization.')
    sample.extend(random.sample(pool, n_need))
random.shuffle(sample)
sample_id = [d['id'] for d in sample]
for index in range(len(data)):
    if data[index]['id'] not in sample_id and 'personalization' in data[index]:
        del data[index]['personalization']
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print('Done →', OUTPUT_FILE)
'''

##### 为训练集分配persona，通用--一半一致一半不一致；个性--一致
# '''
filename = 'megascience'
with open(f'./train-dataprocess/sft/data/{filename}.json', 'r') as f:
    data = json.load(f)
n = len([d for d in data if 'personalization' not in d])
print(n)
idlist = [0] * int(n/2) + [1] * int(n/2)
i = 0
random.shuffle(idlist)
print(idlist)
for index in range(len(data)):
    if 'personalization' in data[index]:
        data[index]['persona'] = data[index]['aligned_persona']
    else:
        data[index]['persona'] = data[index]['aligned_persona'] if idlist[i] == 1 else data[index]['unaligned_persona']
        i += 1
print(i)
with open(f'./train-dataprocess/sft/data/{filename}.json', 'w') as f:
    f.write(json.dumps(data, ensure_ascii=False, indent=4))
# '''

#####
# filename = 'AlignX'
# with open(f'./train-dataprocess/sft/data/{filename}.json', 'r') as f:
#     data = json.load(f)
# for index in range(len(data)):
#     # data[index]['persona'] = data[index]['persona'][0].lower() + data[index]['persona'][1:]
#     if 'general' in data[index]:
#         del data[index]['general']
#     if 'personalization' in data[index]:
#         del data[index]['personalization']
#     del data[index]['persona']
# with open(f'./train-dataprocess/sft/data/{filename}.json', 'w') as f:
#     f.write(json.dumps(data, ensure_ascii=False, indent=4))