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

key_dic = {
    "AlignX": 'answer',
    "UltraMedical":"answer",
    "flan":"answer",
    "megascience": "judgement"
}

def extract_res(text, taskname):
    key = key_dic[taskname]
    if not text:
        return None
    try:
        res = json.loads(text).get(key)
    except json.decoder.JSONDecodeError:
        try:
            res = json.loads(text.replace('```json":', '').replace('```json', '').replace('```', '').replace('\n', ' ')).get(key, None)
        except json.decoder.JSONDecodeError:
            # print(text)
            return None
    return res

def acc(result, taskname):
    if taskname == 'megascience':
        accuracy = sum([1 if d['extract_answer'] == 'correct' else 0 for d in result]) / len(result)
    else:
        accuracy = sum([1 if d['a'] == d['extract_answer'] else 0 for d in result]) / len(result)
    return accuracy

def true_data(result, taskname):
    rightcase = [d for d in result if d['extract_answer'] == 'correct']
    # rightcase = [d for d in result if d['a'] == d['extract_answer']]
    for index in range(len(rightcase)):
        try:
            rightcase[index]['reasoning'] = json.loads(rightcase[index]['answer']).get('thinking')
            # rightcase[index]['reasoning'] = json.loads(rightcase[index]['extractor_answer']).get('thinking')
        except TypeError:
            rightcase[index]['reasoning'] = None
        except json.decoder.JSONDecodeError:
            rightcase[index]['reasoning'] = None
    return rightcase

def whetherin(a_list, a):
    if not a:
        return False
    elif type(a) in (int, float):
        return True if any(str(w) == str(a) for w in a_list) else False
    else:    
        return True if any(str(w) in str(a) for w in a_list) else False

def eval(taskname, filename):
    ## load result data
    with open(f"./train-dataprocess/sft/extractor/{filename}.jsonl", "r") as f:
        result = [json.loads(line) for line in f if line.strip()]
    ## extract answer
    for index in range(len(result)):
        result[index]['extract_answer'] = extract_res(result[index]['extractor_answer'], taskname)
    ## caculate metrics
    accuracy = acc(result, taskname)
    print(f"Data size = {len(result)}, None Data size = {len([d for d in result if not d['extract_answer']])}")
    print(f'Accuracy = {accuracy}')
    rightcase = true_data(result, taskname)
    return rightcase

filename = 'megascience_aligned'
taskname = filename.split('_')[0]
rightcase = eval(taskname, filename)
rightcase_id = [d['id'] for d in rightcase]
# rightcase_id = [d['id'] for d in rightcase if d['reasoning']]
nn = filename.split('_')[0]
with open(f'./train-dataprocess/sft/data/{nn}.json', 'r') as f:
    data = json.load(f)
for index in range(len(data)):
    if data[index]['id'] in rightcase_id:
        data[index]['personalization'] = ''
with open(f'./train-dataprocess/sft/data/{nn}2.json', 'w') as f:
    f.write(json.dumps(data, ensure_ascii=False, indent=4))