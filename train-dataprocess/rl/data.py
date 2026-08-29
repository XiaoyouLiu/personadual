import re
import os
import pandas as pd
import numpy as np
import json
import tqdm
import yaml
import random
import pyarrow.parquet as pq
from pathlib import Path
import ijson
from collections import defaultdict, Counter

####### formulate RL data
'''
with open('./train-dataprocess/rl/used_data/AlignX2.json', 'r') as f:
    alignx = json.load(f)
with open('./train-dataprocess/rl/used_data/flan2.json', 'r') as f:
    flan = json.load(f)
with open('./train-dataprocess/rl/used_data/UltraMedical2.json', 'r') as f:
    ultra = json.load(f)
data = []
for d in alignx:
    p = d['aligned_persona']
    dd = {
        "data_source": 'AlignX',
        "prompt": [{
            "role": "user",
            "content": f"I am {p}, my question is:\n{d['q']}\nPlease return your answer in JSON format:\n" + """{"answer":"A/B/..."}""",
        }],
        "reward_model":{
            "style": "rule",
            "ground_truth": d['a']
        }
    }
    data.append(dd)
for d in flan:
    p = d['aligned_persona'] if 'aligned_persona' in d else d['unaligned_persona']
    if d.get('task', '') in ["math_dataset_10templates", "natural_questions_10templates"]:
        dd = {
            "data_source": 'flan',
            "prompt": [{
                "role": "user",
                "content": f"{d['q']}\nPlease return your answer in JSON format:\n" + """{"answer":"your answer"}""",
            }],
            "reward_model":{
                "style": "rule",
                "ground_truth": d['a']
            }
        }
    else:
        dd = {
            "data_source": 'flan',
            "prompt": [{
                "role": "user",
                "content": f"{d['q']}\nPlease return your answer in JSON format:\n" + """{"answer":"A/B/..."}""",
            }],
            "reward_model":{
                "style": "rule",
                "ground_truth": d['a']
            }
        }
    data.append(dd)
for d in ultra:
    p = d['aligned_persona'] if 'aligned_persona' in d else d['unaligned_persona']
    dd = {
        "data_source": 'UltraMedical',
        "prompt": [{
            "role": "user",
            "content": f"{d['q']}\nPlease return your answer in JSON format:\n" + """{"answer":"A/B/..."}""",
        }],
        "reward_model":{
            "style": "rule",
            "ground_truth": d['a']
        }
    }
    data.append(dd)
print(len(data))
data2 = [data[i] for i in random.sample(range(len(data)), 500)]
# data = [data[i] for i in random.sample(range(len(data)), 5000)]
df = pd.DataFrame(data)
df2 = pd.DataFrame(data2)
df.to_parquet("./rl/rl-data/train-think.parquet", index=False) 
df2.to_parquet("./rl/rl-data/val-think.parquet", index=False) 
'''
'''
with open('./train-dataprocess/rl/used_data/AlignX2.json', 'r') as f:
    alignx = json.load(f)
with open('./train-dataprocess/rl/used_data/flan2.json', 'r') as f:
    flan = json.load(f)
with open('./train-dataprocess/rl/used_data/UltraMedical2.json', 'r') as f:
    ultra = json.load(f)
data = []
for d in alignx:
    p = d['aligned_persona']
    dd = {
        "data_source": 'AlignX',
        "prompt": [{
            "role": "user",
            "content": f"I am {p}, my question is:\n{d['q']}\nPlease return your answer in JSON format:\n" + """{"answer":"A/B/..."}""",
        }],
        "reward_model":{
            "style": "rule",
            "ground_truth": d['a']
        },
        "extra_info": {
                'persona': p,
                'question': d['q']
        }    
    }
    data.append(dd)
for d in flan:
    p = d['aligned_persona'] if 'aligned_persona' in d else d['unaligned_persona']
    if d.get('task', '') in ["math_dataset_10templates", "natural_questions_10templates"]:
        dd = {
            "data_source": 'flan',
            "prompt": [{
                "role": "user",
                "content": f"I am {p}, my question is:\n{d['q']}\nPlease return your answer in JSON format:\n" + """{"answer":"your answer"}""",
            }],
            "reward_model":{
                "style": "rule",
                "ground_truth": d['a']
            },
            "extra_info": {
                'persona': p,
                'question': d['q']
            }    
        }
    else:
        dd = {
            "data_source": 'flan',
            "prompt": [{
                "role": "user",
                "content": f"I am {p}, my question is:\n{d['q']}\nPlease return your answer in JSON format:\n" + """{"answer":"A/B/..."}""",
            }],
            "reward_model":{
                "style": "rule",
                "ground_truth": d['a']
            },
            "extra_info": {
                'persona': p,
                'question': d['q']
            }    
        }
    data.append(dd)
for d in ultra:
    p = d['aligned_persona'] if 'aligned_persona' in d else d['unaligned_persona']
    dd = {
        "data_source": 'UltraMedical',
        "prompt": [{
            "role": "user",
            "content": f"I am {p}, my question is:\n{d['q']}\nPlease return your answer in JSON format:\n" + """{"answer":"A/B/..."}""",
        }],
        "reward_model":{
            "style": "rule",
            "ground_truth": d['a']
        },
        "extra_info": {
                'persona': p,
                'question': d['q']
        }    
    }
    data.append(dd)
print(len(data))
data2 = [data[i] for i in random.sample(range(len(data)), 500)]
# data = [data[i] for i in random.sample(range(len(data)), 5000)]
df = pd.DataFrame(data)
df2 = pd.DataFrame(data2)
df.to_parquet("./rl/rl-data/train.parquet", index=False) 
df2.to_parquet("./rl/rl-data/val.parquet", index=False) 
'''
'''
with open('./train-dataprocess/rl/used_data/AlignX2.json', 'r') as f:
    alignx = json.load(f)
with open('./train-dataprocess/rl/used_data/flan2.json', 'r') as f:
    flan = json.load(f)
with open('./train-dataprocess/rl/used_data/UltraMedical2.json', 'r') as f:
    ultra = json.load(f)
data1, data2 = [], []
for d in alignx:
    p = d['aligned_persona']
    dd = {
        "data_source": 'AlignX',
        "prompt": [{
            "role": "user",
            "content": f"I am {p}, my question is:\n{d['q']}\nPlease return your answer in JSON format:\n" + """{"answer":"A/B/..."}""",
        }],
        "reward_model":{
            "style": "rule",
            "ground_truth": d['a']
        }
    }
    data1.append(dd)
for d in flan:
    p = d['aligned_persona'] if 'aligned_persona' in d else d['unaligned_persona']
    if d.get('task', '') in ["math_dataset_10templates", "natural_questions_10templates"]:
        dd = {
            "data_source": 'flan',
            "prompt": [{
                "role": "user",
                "content": f"{d['q']}\nPlease return your answer in JSON format:\n" + """{"answer":"your answer"}""",
            }],
            "reward_model":{
                "style": "rule",
                "ground_truth": d['a']
            }
        }
    else:
        dd = {
            "data_source": 'flan',
            "prompt": [{
                "role": "user",
                "content": f"{d['q']}\nPlease return your answer in JSON format:\n" + """{"answer":"A/B/..."}""",
            }],
            "reward_model":{
                "style": "rule",
                "ground_truth": d['a']
            }
        }
    data2.append(dd)
for d in ultra:
    p = d['aligned_persona'] if 'aligned_persona' in d else d['unaligned_persona']
    dd = {
        "data_source": 'UltraMedical',
        "prompt": [{
            "role": "user",
            "content": f"{d['q']}\nPlease return your answer in JSON format:\n" + """{"answer":"A/B/..."}""",
        }],
        "reward_model":{
            "style": "rule",
            "ground_truth": d['a']
        }
    }
    data2.append(dd)
data1_test = [data1[i] for i in random.sample(range(len(data1)), 100)]
data2_test = [data2[i] for i in random.sample(range(len(data2)), 400)]
df1 = pd.DataFrame(data1)
df1_test = pd.DataFrame(data1_test)
df2 = pd.DataFrame(data2)
df2_test = pd.DataFrame(data2_test)
print(len(df1), len(df1_test), len(df2), len(df2_test))
df1.to_parquet("./rl/rl-data/train-personal.parquet", index=False) 
df1_test.to_parquet("./rl/rl-data/val-personal.parquet", index=False) 
df2.to_parquet("./rl/rl-data/train-general.parquet", index=False) 
df2_test.to_parquet("./rl/rl-data/val-general.parquet", index=False) 
'''
####### add persona
'''
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
taskname = 'UltraMedical-rl'
with open(f'./train-dataprocess/rl/persona_gen/{taskname}_Alignedpersona.jsonl', 'r') as f:
    data = [json.loads(line) for line in f if line.strip()]
persona = {}
for index in range(len(data)):
    p, _ = extract_outermost_json(data[index]['answer'])
    persona[str(data[index]['id'])] = p.get('persona', '') if p else None
taskname = 'UltraMedical'
with open(f'./train-dataprocess/rl/used_data/{taskname}2.json', 'r') as f:
    data = json.load(f)
for index in range(len(data)):
    if str(data[index]['id']) in persona:
        data[index]['aligned_persona'] = persona[str(data[index]['id'])] 
with open(f'./train-dataprocess/rl/used_data/{taskname}2.json', 'w') as f:
    f.write(json.dumps(data, ensure_ascii=False, indent=4))
'''
####### 抽样RL数据
'''
# flan
with open(f'./train-dataprocess/rl/data/flan.json', 'r') as f:
    data = json.load(f)
print(len(data))
TOTAL_WANT = 2000            # 固定总抽取量
seed = 42
src2idx = defaultdict(list)
for idx, item in enumerate(data):
    src2idx[item['task']].append(idx)
total_pop = len(data)                # 总人数
src_quota = {}                       # 每个来源分到的名额
leftover  = TOTAL_WANT               # 剩余未分配名额
for src, idx_list in src2idx.items():
    quota = len(idx_list) / total_pop * TOTAL_WANT
    int_part = int(quota)
    src_quota[src] = int_part
    leftover -= int_part
for src, _ in sorted(src_quota.items(),
                     key=lambda x: (len(src2idx[x[0]])/total_pop*TOTAL_WANT) % 1,
                     reverse=True):
    if leftover <= 0:
        break
    src_quota[src] += 1
    leftover -= 1
sampled_idx = set()
for src, quota in src_quota.items():
    idx_list = src2idx[src]
    pick = min(quota, len(idx_list))      # 防止来源本身不足
    sampled_idx.update(random.sample(idx_list, pick))
print(f'实际抽中 {len(sampled_idx)} 条（目标 {TOTAL_WANT}）')
sampled, remain = [], []
for idx, item in enumerate(data):
    (sampled if idx in sampled_idx else remain).append(item)
flan = sampled

# ALignX
with open(f'./train-dataprocess/rl/data/AlignX.json', 'r') as f:
    alignx = json.load(f)
alignx_id = random.sample(range(len(alignx)), 500)
alignx = [alignx[i] for i in alignx_id]

#ultramedical
with open(f'./train-dataprocess/rl/data/UltraMedical.json', 'r') as f:
    data = json.load(f)
k = 250
source2idx = defaultdict(list)
for idx, item in enumerate(data):
    src = item['task_id'].split(',', 1)[0]
    source2idx[src].append(idx)
sampled_idx = set()
for src, idx_list in source2idx.items():
    pick = min(k, len(idx_list))
    sampled_idx.update(random.sample(idx_list, pick))
result = []
for idx, item in enumerate(data):
    if idx in sampled_idx:
        result.append(item)
ultra = result
'''
### add persona
'''
with open('./train-dataprocess/rl/used_data/flan2.json', 'r') as f:
    flan = json.load(f)
flan1, flan = [d for d in flan if 'aligned_persona' in d or 'unaligned_persona' in d], [d for d in flan if 'aligned_persona' not in d and 'unaligned_persona' not in d]
with open('./train-dataprocess/rl/used_data/UltraMedical2.json', 'r') as f:
    ultra = json.load(f)
ultra1, ultra = [d for d in ultra if 'aligned_persona' in d or 'unaligned_persona' in d], [d for d in ultra if 'aligned_persona' not in d and 'unaligned_persona' not in d]
print(len(flan), len(ultra))
with open(f'./train-dataprocess/data/persona/personahub/persona.jsonl', 'r') as f:
    persona = [json.loads(line) for line in f if line.strip()]
    sampled_persona = [persona[i] for i in random.sample(range(len(persona)), 2999)]
idlist = [0] * 1999 + [1] * 1999
random.shuffle(idlist)
idx = 0
for index in range(len(flan)):
    if idlist[index] == 1:
        p = re.sub(r'(?i)^I\s*am\s*|^I[\u0027\u2019]m\s*|^我是\s*', '', sampled_persona[idx]['persona']).strip()
        idx += 1
        flan[index]['unaligned_persona'] = p[:1].lower() + p[1:]
idlist = [0] * 1000 + [1] * 1000
random.shuffle(idlist)
for index in range(len(ultra)):
    if idlist[index] == 1:
        p = re.sub(r'(?i)^I\s*am\s*|^I[\u0027\u2019]m\s*|^我是\s*', '', sampled_persona[idx]['persona']).strip()
        idx += 1
        ultra[index]['unaligned_persona'] = p[:1].lower() + p[1:]

with open(f'./train-dataprocess/rl/used_data/flan3.json', 'w') as f:
    f.write(json.dumps(flan1+flan, ensure_ascii=False, indent=4))
with open(f'./train-dataprocess/rl/used_data/UltraMedical3.json', 'w') as f:
    f.write(json.dumps(ultra1+ultra, ensure_ascii=False, indent=4))
# with open(f'./train-dataprocess/rl/used_data/AlignX.json', 'w') as f:
#     f.write(json.dumps(alignx, ensure_ascii=False, indent=4))
'''