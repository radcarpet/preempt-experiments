import re
import names
import math
import numpy as np
from pyfpe_ff3 import format_align_digits

import sys
sys.path.insert(0, '../universal-ner')
from src.utils import *
sys.path.insert(0, '../privacypromptrewriting')
from utils import format_align_digits
from transformers import AutoTokenizer, AutoModelForCausalLM, MarianMTModel
import torch
from torch.utils.data import Dataset, DataLoader
import gc
from openai import OpenAI

from tqdm.auto import tqdm
import asyncio
import ast
import json
import traceback 
import time

import random

import unicodedata

from transformers import logging
import os

from fastchat.conversation import get_conv_template

logging.set_verbosity_error()


"""
Pretty printing
"""
def print_block(vals):
    for val in vals:
        pprint(val)
    print('#'*30)

def pprint(tag, val):
    if isinstance(val, str):
        print(f"{tag} {'.'*(30 - len(val) -len(tag))} {val}")
    else:
        print(f"{tag} {'.'*(30 - 5 -len(tag))} {val:.3f}")

"""
Data loading and preprocessing
"""
def load_json_dataset(data_path):
    with open(data_path, 'r') as fp:
        data = json.load(fp)
    return data

### json saver
def save_json_list(data, path):
    with open(path, 'w') as fp:
        json.dump(data, fp, indent=2)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

### Dataloader for pipelining
class any2en(Dataset):
    def __init__(self, text):
        self.text = text

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx]


"""
Preprocessing and postprocessing helpers
"""
def clean_suffix_name_QA(all_text):
    outputs = all_text
    suffixes = ['\"', '"', '[', ']', '\\ ']
    for suffix in suffixes:
        outputs = [output.replace(suffix, '') for output in outputs]
    return outputs

def clean_suffix_name(all_text):
    outputs = all_text
    suffixes = ['Prof ', 'Prof.', 'Prof', 'Mrs. ', 'Mrs ', 'Mrs.', 'Mrs', 'Mr ', 'Mr. ', 'Mr.', 'Mr', 'Ms. ', 'Ms.', 'Ms', 'Mrs ', 'Mrs', 'Ms ', 'Ms', 'Herr ', 'Herrn ', 'Frau ', 'M. ', 'Mme. ', 'M ', 'Mme ', 'Madame ', 'Monsieur ', 'Monsieur ', '\"', '"', '[', ']', '\\ ', 'Dr. ', 'Dr.', 'Miss ', 'Miss']
    for suffix in suffixes:
        outputs = [output.replace(suffix, '') for output in outputs]
    outputs = ['"[' + output.replace("'","") + ']"' for output in outputs]
    return outputs

def postprocess_output_QA(outputs, output_dict, entity_type):
    if entity_type=="Name" or entity_type=="Full Name":
        outputs = clean_suffix_name_QA(outputs) 

    for output in outputs:
        output_dict[entity_type].append(output)

    return output_dict

def postprocess_output(outputs, output_dict, entity_type):
    if entity_type=="Name" or entity_type=="Full Name":
        outputs = clean_suffix_name(outputs) 
    if entity_type=="Age":
        outputs = [str(re.findall(r"\b(\d{1,3})\b", output)) for output in outputs]
    if entity_type=="Zipcode":
        outputs = [str(re.findall(r"\b(\d{5}(?:-\d{4})?)\b", output)) for output in outputs]
    if entity_type=="Money":
        outputs = re.findall(r"[-+]?(?:\d*\.*\,*\s*\d+)", outputs[0])
        outputs = [output.strip() for output in outputs]
    for output in outputs:
        # print(outputs)
        if len(output) > 0: 
            if entity_type=="Money":
                output_dict[entity_type].append(outputs)
                return output_dict
            elif entity_type=="Name" or entity_type=="Full Name":
                temp = unicodedata.normalize("NFKD", ast.literal_eval(output))
                temp = temp.replace('[', '')
                temp = temp.replace(']', '')
                temp = temp.split(", ")
                output_dict[entity_type].append(temp)
            else:
                output_dict[entity_type].append(ast.literal_eval(output))

    if len(outputs)==0:
        outputs.append(None)
        output_dict[entity_type].append(outputs)
    return output_dict


def postprocess_output_ner(outputs, output_dict, entity_type):
    if entity_type=="Age":
        outputs = [str(re.findall(r"\b(\d{1,3})\b", output)) for output in outputs]
    if entity_type=="Zipcode":
        outputs = [str(re.findall(r"\b(\d{5}(?:-\d{4})?)\b", output)) for output in outputs]
    if entity_type=="Money":
        outputs = [output.replace(",", "") for output in outputs]
        outputs = [str(re.findall(r"[-+]?(?:\d*\.*\d+)", output)) for output in outputs]

    temp = []
    for output in outputs:
        if entity_type=="Age":
            output = ast.literal_eval(output)
            temp.extend(output)
        elif entity_type=="Money":
            temp = ast.literal_eval(output)
            temp = [t.replace('-','') for t in temp]
        else:
            temp.append(output)

    output_dict[entity_type].append(temp)
    return output_dict

def postprocess_output_gpt(outputs, output_dict, entity_type):
    if entity_type=="Name" or entity_type=="Full Name":
        outputs = clean_suffix_name(outputs) 
    if entity_type=="Age":
        outputs = [str(re.findall(r"\b(\d{1,3})\b", output)) for output in outputs]
    if entity_type=="Zipcode":
        outputs = [str(re.findall(r"\b(\d{5}(?:-\d{4})?)\b", output)) for output in outputs]
    if entity_type=="Money":
        outputs = [output.replace(",", "") for output in outputs]
        outputs = [str(re.findall(r"[-+]?(?:\d*\.*\d+)", output)) for output in outputs]

    temp = []
    for output in outputs:
        if len(output) > 0: 
            if entity_type=="Name" or entity_type=="Full Name":
                _temp = unicodedata.normalize("NFKD", ast.literal_eval(output))
                _temp = _temp.replace('[', '')
                _temp = _temp.replace(']', '')
                _temp = _temp.split(", ")
                temp.extend(_temp)
            elif entity_type=="Age":
                output = ast.literal_eval(output)
                temp.extend(output)
            elif entity_type=="Money":
                temp = ast.literal_eval(output)
                temp = [t.replace('-','') for t in temp]
            else:
                temp.append(output)

    output_dict[entity_type].append(temp)
    return output_dict

def find_entity(data, entity, lang, samples):
    texts = []
    values = []
    counter = 0
    for datum in data:
        if lang!=datum['lang'] or entity not in datum['labels']: continue
        texts.append(datum['conversations'][0]['value'].replace('Passage: ','').split('\n\n')[0])
        values.append(datum['label_values'][entity])
        counter += 1
        if counter==samples: break

    return texts, values

def gpt_cleaner(queries, entity, samples=-1):
    queries = queries[0][:samples]
    t_dict = {entity: []}
    for query in queries:
        query = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", r"\\'", query)
        if 'It seems' in query or 'cannot' in query: 
            t_dict[entity].append([])
            continue
        query = ast.literal_eval(query)[entity]
        t_dict = postprocess_output_gpt(query, t_dict, entity)

    return t_dict

def gpt_cleaner(queries, entity, samples=-1):
    queries = queries[:samples]
    t_dict = {entity: []}
    for query in queries:
        query = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", r"\\'", query)
        if 'It seems' in query or 'cannot' in query: 
            t_dict[entity].append([])
            continue
        query = ast.literal_eval(query)[entity]
        t_dict = postprocess_output_gpt(query, t_dict, entity)

    return t_dict

def llama3_cleaner(queries, entity, samples=-1):
    queries = queries[entity]
    temp_dict = {entity: []}
    if entity=='Name':
        for query in queries:
            temp = []
            for qq in query:
                qq = qq.replace("'","")
                temp.append(qq)

            temp_dict[entity].append(temp)
    else:
        for query in queries:
            temp_dict[entity].append(query)

    return temp_dict

def gemini_cleaner(queries, entity='Age', samples=-1):
    query = queries
    print("QUERIES", queries)
    t_dict = {entity: []}
    query = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", r"\\'", query)
    query = query.replace('```','')
    query = query.replace('\n', '')
    query = query.replace('python','')
    query = query.replace('json','')
    query = query.replace('[[', '[')
    query = query.replace(']]', ']')
    query = query.replace('}]}', '}')
    if entity=='Money':
        query = query.replace('b', '')
        query = query.replace('m', '')
        query = query.replace('k', '')

    try:
        query = ast.literal_eval(query)[entity]
    except:
        query = ast.literal_eval(query)
#     print(query)
    if entity=='Money':
        t_dict[entity] = query
        temp = []
        for query in t_dict[entity]:
            temp.append(query.replace(',', ''))
            t_dict[entity] = temp
        return t_dict

    t_dict = postprocess_output_gpt(query, t_dict, entity)
    t_dict[entity] = t_dict[entity][0]

    return t_dict

def gemini_cleaner_list(queries, entity, samples=-1):
    queries = queries[0]
    t_dict = {entity: []}
    for query in queries:
        query = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", r"\\'", query)
        query = query.replace('```','')
        query = query.replace('json','')
        if 'It seems' in query or 'cannot' in query: 
            t_dict[entity].append([])
            continue
        query = ast.literal_eval(query)[entity]
        t_dict = postprocess_output_gpt(query, t_dict, entity)

    return t_dict


"""
NER experiment helpers
"""
def ner_cleaner(queries, entity, lang, samples=-1):
    # Return a list of lists!
    queries = queries[lang][entity][:samples]
    t_dict = {entity: []}
    for query in queries: 
        # query is a list.
        t_dict = postprocess_output_ner(query, t_dict, entity)

    return t_dict[entity]

def ner_match_score(queries, answers, entity, samples=-1):
    queries = queries[entity]
    tp = 0
    fn = 0
    fp = 0
    for query, answer in zip(queries[:samples], answers[:samples]):
        a = set(query)
        b = set(answer)
        c = a.intersection(b)
        tp += len(c)
        fn += len(b) - len(c)
        fp += len(a) - len(c)

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    return precision, recall, tp, fn, fp


def ner_match_score_all(queries, answers, entity, samples=-1):
    queries = queries[entity]
    tp = 0
    fn = 0
    fp = 0

    all_queries = []
    all_answers = []

    for query in queries[:samples]:
        all_queries.extend(query)

    for answer in answers[:samples]:
        all_answers.extend(answer)

    a = set(all_queries)
    b = set(all_answers)
    c = a.intersection(b)
    tp += len(c)
    fn += len(b) - len(c)
    fp += len(a) - len(c)

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    return precision, recall, tp, fn, fp


def ner_match_score_gemini(queries, answers, entity, samples=-1):
    queries = queries[entity]
    tp = 0
    fn = 0
    fp = 0

    all_queries = []
    all_answers = []

    for query in queries:
        indices = [i for i, x in enumerate(queries) if x == query]
        indices = list(range(len(indices)))
        for index in indices:
            all_queries.append(f"{query}_{index}")


    temp_answers = []
    for answer in answers[:samples]:
        temp_answers.extend(answer)

    for answer in temp_answers:
        indices = [i for i, x in enumerate(temp_answers) if x == answer]
        indices = list(range(len(indices)))
        for index in indices:
            all_answers.append(f"{answer}_{index}")

    a = set(all_queries)
    b = set(all_answers)
    c = a.intersection(b)
    tp += len(c)
    fn += len(b) - len(c)
    fp += len(a) - len(c)

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    return precision, recall, tp, fn, fp


"""
NER Extraction methods
"""
# GPT-4 NER
def extract_entities_GPT(all_text, entity, api_key, list_of_entities=['Name','Age','Money','Zipcode','SSN','Date']):
#     entities = ", ".join(list_of_entities)
    client = OpenAI(api_key=api_key,)
    async def query_llm(text, model="gpt-4-turbo"):
        response = client.chat.completions.create(model=model,
        messages=[
          {
            "role": "system",
              # Prompt format from Ye et al. https://arxiv.org/pdf/2303.10420.pdf
            "content": f"Please identify {entity} from the given text. Format the output as a dictionary of lists: {{'{entity}': ['{entity}_1', '{entity}_2']}}"
          },
          {
            "role": "user",
            "content": text 
          },
        ],
        temperature=0,
        max_tokens=4096,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)
        return response

    async def extract_text(text, idx, crash_array):
        try:
            response = await query_llm(text)
            response = response.choices[0].message.content
            response = response.encode().decode()
            return response
        except Exception as e:
            print(f"Timed out: Index {idx}. Retrying...")
            crash_array.append(idx)
            response = await extract_text(text, idx, crash_array)
            return response

    crash_arrays = []
    loop = asyncio.get_event_loop()
    arr = all_text
    counter = range(len(arr))
    crash_array = []
    a = time.time()
    print(f"Starting API NER extraction. Time: {a}")
    translated_de_encrypted_en = loop.run_until_complete(asyncio.gather(*[extract_text(arr[i], i, crash_array) for i in counter],
                                                      return_exceptions=True))
    print(f"Task target:{len(arr)} Time taken: {time.time()-a}")
    crash_arrays.append(crash_array)
    print(f"Crash arrays:", crash_arrays)

    return translated_de_encrypted_en, entity

# Claude NER
def claude_cleaner(content, lang, entity):
    if 'Sorry' in content or 'no' in content or 'apologize' in content:
        return []
    values = content.split(":")
    if len(values)==1: return []
    values = values[1]
    values = values.split("\n\n")[1]
    values = values.split('\n')
    temp = []
    for value in values:
        if entity=='Name':
            pattern = r'[a-zA-Z]+'
            value = re.findall(pattern, value)
            value = " ".join(value)
        if entity=='Money' or entity=='Age':
            pattern = r'[a-zA-Z]*(\d{1,10}(?:,\d{3})*(?:\.\d+)?)[a-zA-Z]*'
            value = re.findall(pattern, value)
            if len(value)==0:
                value = ""
            else:
                value = value[-1]
                value = value.replace(',','')

        temp.append(value)

    return temp

def load_claude_data(lang, entity):
    temp = []
    for i in range(1, 301):
        path = f'datasets/claude/{lang}_{entity}/{i}.txt' if lang!='english' or entity!='Name' else f'datasets/claude/{lang}_{entity}/{lang}_{entity}_test_{i}.txt'
        if os.path.exists(path):
            data = load_json_dataset(path)
            temp.append(claude_cleaner(data['content'][0]['text'], lang, entity))
        else:
            temp.append([])

    return {entity: temp}

def extract_entities_claude(all_text, entity, api_key, list_of_entities=['Name','Age','Money','Zipcode','SSN','Date']):
    import anthropic

    async def query_llm(text, model="claude-3-opus-20240229"):
        client = anthropic.Anthropic(
            # defaults to os.environ.get("ANTHROPIC_API_KEY")
            api_key=api_key,
        )
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            system = f"Please identify {entity} from the given text. Format the output as a dictionary of lists: {{'{entity}': ['{entity}_1', '{entity}_2']}}",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            ]
        )
        response = message.content
        return response

    async def extract_text(text, idx, crash_array):
        try:
            response = await query_llm(text)
            response = response.choices[0].message.content
            response = response.encode().decode()
            return response
        except Exception as e:
            print(f"Timed out: Index {idx}. Retrying...")
            crash_array.append(idx)
            response = await extract_text(text, idx, crash_array)
            return response

    crash_arrays = []
    loop = asyncio.get_event_loop()
    arr = all_text
    counter = range(len(arr))
    crash_array = []
    a = time.time()
    print(f"Starting API NER extraction. Time: {a}")
    translated_de_encrypted_en = loop.run_until_complete(asyncio.gather(*[extract_text(arr[i], i, crash_array) for i in counter],
                                                      return_exceptions=True))
    print(f"Task target:{len(arr)} Time taken: {time.time()-a}")
    crash_arrays.append(crash_array)
    print(f"Crash arrays:", crash_arrays)

    return translated_de_encrypted_en, entity

# Gemini NER
def extract_entities_gemini(all_text, entity, lang, api_key, list_of_entities=['Name','Age','Money','Zipcode','SSN','Date']):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    generation_config = genai.types.GenerationConfig(
        temperature=0,
        max_output_tokens=4096,
        top_p=1,
    )

    sleep_time = 600
    def query_llm(text, model="gemini-1.5-pro-001"):
        model = genai.GenerativeModel(
            model_name = model,
            generation_config=generation_config,
            system_instruction=[
                f"Please identify {entity} from the given text. Format the output as a dictionary of lists: {{'{entity}': ['{entity}_1', '{entity}_2']}}." if entity!='Money' else f"Please identify {entity} from the given text. Do not provide the currency, only provide the value. Format the output as a dictionary of lists: {{'{entity}': ['{entity}_1', '{entity}_2']}}."
            ]
        )

        temp_text = text
        response = model.generate_content(temp_text)
        text = response._result.candidates[0].content.parts[0].text
#         text = response
        return text

    crash_arrays = []
    loop = asyncio.get_event_loop()
    arr = all_text
    counter = range(len(arr))
    crash_array = []
    a = time.time()
    print(f"Starting API NER extraction. Time: {a}")
    translated_de_encrypted_en = query_llm(arr)

    print(f"Task target:{len(arr)} Time taken: {time.time()-a}")
    crash_arrays.append(crash_array)
    print(f"Crash arrays:", crash_arrays)

    return translated_de_encrypted_en, entity

def extract_entities_gemini_async(all_text, entity, lang, api_key, list_of_entities=['Name','Age','Money','Zipcode','SSN','Date']):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    generation_config = genai.types.GenerationConfig(
        temperature=0,
        max_output_tokens=4096,
        top_p=1,
    )

    sleep_time = 600
    async def query_llm(text, model="gemini-1.5-pro-001"):
        model = genai.GenerativeModel(
            model_name = model,
            generation_config=generation_config,
            system_instruction=[
                f"Please identify {entity} from the given text. Format the output as a dictionary of lists: {{'{entity}': ['{entity}_1', '{entity}_2']}}."
            ]
        )

        temp_text = [f"{text}"]
        response = model.generate_content(temp_text)
        text = response._result.candidates[0].content.parts[0].text
        return text

    async def extract_text(text, idx, crash_array, sleep_time=600):
        try:
            response = await query_llm(text)
            return response
        except Exception as e:
            print(e)
            return response

    crash_arrays = []
    loop = asyncio.get_event_loop()
    arr = all_text
    counter = range(len(arr))
    crash_array = []
    a = time.time()
    print(f"Starting API NER extraction. Time: {a}")
    translated_de_encrypted_en = loop.run_until_complete(asyncio.gather(*[extract_text(arr[i], i, crash_array) for i in counter],return_exceptions=True))

    print(f"Task target:{len(arr)} Time taken: {time.time()-a}")
    crash_arrays.append(crash_array)
    print(f"Crash arrays:", crash_arrays)

    return translated_de_encrypted_en, entity


def extract_entities_gemma(
    all_text,
    model=None,
    tokenizer=None,
    batch_size=1,
    entity='Name',
    device='cuda',
):

    output_dict = {}
    output_dict[entity] = []

    def format_prompt(input):

        chat = [
                { "role": "user", 
                 # Not for gemini: "content": f"Please identify {entity} from the given text. Format the output as list with no additional text. Example: ['{entity} 1', '{entity} 2'].\n\nText: {input}" if entity!='Money' else f"Please identify Currency Value from the given text. Format the output as list with no additional text. Example: ['Currency Value 1', 'Currency Value 2'].\n\nText: {input}"},
                 "content": f"Please identify {entity} from the given text. Format the output as list with no additional text. Example: ['{entity} 1', '{entity} 2'].\n\nText: {input}" if entity!='Money' else f"Please identify the numerical values of Money from the given text. DO NOT ADD '000' TO ANY VALUE. DO NOT REMOVE SPACES IN ANY VALUE. Format the output as list with no additional text. Example: ['Value 1', 'Value 2'].\n\nText: {input}"},
                # { "role": "assistant", "content": "{output}" },
            ]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        # print(text)
        return text

    def collate_fn(batch):
        tokenized_text = tokenizer(batch,  
                                   return_tensors='pt', 
                                   add_special_tokens=False
                                  )
        # print(tokenized_text)
        return tokenized_text

    model.eval()
    max_new_tokens=4096

    prompts = [format_prompt(text) for text in all_text]
    prompts = any2en(prompts)
    prompt_dataloader = DataLoader(prompts, collate_fn=collate_fn, batch_size=batch_size)

    for i, prompt_batch in tqdm(enumerate(prompt_dataloader), total=len(prompt_dataloader)):
        with torch.no_grad():
            prompt_batch.to(device)
            outputs = model.generate(**prompt_batch, 
                                     do_sample=False,
                                     max_length=max_new_tokens)
            outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            outputs = [output.replace("['", "") for output in outputs]
            outputs = [output.replace("']", "") for output in outputs]
            outputs = [r.split('\nmodel\n')[-1].strip() for r in outputs]
            output_dict = postprocess_output(outputs, output_dict, entity)

    return output_dict

def extract_entities_gemma_sp(
    all_text,
    model,
    tokenizer,
    batch_size,
    entity_type,
    device,
):

    max_new_tokens = 4096
    output_dict = {}
    output_dict[entity_type] = []
    all_text_frags = [text.split('.') for text in all_text]
    temp = []
    bad_guesses = ["Mr"," Mr", "Ms", " Ms", "Mrs", " Mrs", "They", "they", " They", " they", "A", "He", "She", " ", "", "I", "My", "the", "The"]
    for text_frag in all_text_frags:
        t_temp = []
        for frag in text_frag:
            if frag in bad_guesses:
                continue
            t_temp.append(frag)
        temp.append(t_temp)
    all_text_frags = temp

    def format_prompt(input):

        chat = [
                { 
                    "role": "user", 
                    "content": f"Please identify {entity_type} from the given text. Format the output as list with no additional text. Example: ['{entity_type} 1', '{entity_type} 2'].\n\nText: {input}"
                 },
                # { "role": "assistant", "content": "{output}" },
            ]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        # print(text)
        return text

    def collate_fn(batch):
        tokenized_text = tokenizer(batch,  
                                   return_tensors='pt', 
                                   add_special_tokens=False
                                  )
        # print(tokenized_text)
        return tokenized_text

    model.eval()
    max_new_tokens=4096

    for all_text_frag in all_text_frags:
        frag_output_dict = {}
        frag_output_dict[entity_type] = []

        prompts = [format_prompt(text) for text in all_text_frag]
        prompts = any2en(prompts)
        prompt_dataloader = DataLoader(prompts, collate_fn=collate_fn, batch_size=batch_size)

        for i, prompt_batch in tqdm(enumerate(prompt_dataloader), total=len(prompt_dataloader)):
            with torch.no_grad():
                prompt_batch.to(device)
                outputs = model.generate(**prompt_batch, 
                                        do_sample=False,
                                        max_length=max_new_tokens)
                outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                outputs = [output.replace("['", "") for output in outputs]
                outputs = [output.replace("']", "") for output in outputs]
                outputs = [r.split('\nmodel\n')[-1].strip() for r in outputs]
                frag_output_dict = postprocess_output(outputs, frag_output_dict, entity_type)

        temp = []
        for kk in frag_output_dict[entity_type]:
            for kki in kk:
                if kki.strip().replace("'","") not in bad_guesses:
                    temp.append(kki.strip().replace("'",""))
        output_dict[entity_type].append(temp)

    return output_dict

def extract_entities_llama(
    all_text,
    model,
    tokenizer,
    batch_size,
    entity_type,
    device,
):
    entity = entity_type
    output_dict = {}
    output_dict[entity] = []

    def format_prompt(input):
        conv = get_conv_template('llama-3')
        conv.set_system_message(
            f"Please identify that can be categorized as '{entity}'. Format the output as list with no additional text. Example: ['{entity}_1', '{entity}_2']." if entity!='Money' else f"Please identify Currency Value from the given text. Format the output as list with no additional text. Example: ['Currency Value 1', 'Currency Value 2']."
        )
        conv.append_message(conv.roles[0], input)
        conv.append_message(conv.roles[1], None)
        text = conv.get_prompt()
        return text

    def collate_fn(batch):
        tokenized_text = tokenizer(batch,  
                                   return_tensors='pt', 
                                   add_special_tokens=False)
        return tokenized_text

    model.eval()
    max_new_tokens=4096

    prompts = [format_prompt(text) for text in all_text]
    prompts = any2en(prompts)
    prompt_dataloader = DataLoader(prompts, collate_fn=collate_fn, batch_size=batch_size)

    for i, prompt_batch in tqdm(enumerate(prompt_dataloader), total=len(prompt_dataloader)):
        try:
            with torch.no_grad():
                prompt_batch.to(device)
                outputs = model.generate(**prompt_batch, 
                                         do_sample=False,
                                         max_length=max_new_tokens)
                outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                outputs = [output.replace("['", "") for output in outputs]
                outputs = [output.replace("']", "") for output in outputs]
                outputs = [r.split('assistant\n\n')[-1].strip() for r in outputs]
                output_dict = postprocess_output(outputs, output_dict, entity)
        except:
                traceback.print_exc()
                print(f"Error on batch ID: {i}")

    # print(output_dict)
    return output_dict

def extract_entities_llama_sp(
    all_text,
    model,
    tokenizer,
    batch_size,
    entity_type,
    device,
):
    entity = entity_type
    max_new_tokens = 4096
    output_dict = {}
    output_dict[entity] = []
    all_text_frags = [text.split('.') for text in all_text]
    temp = []
    bad_guesses = ["Mr"," Mr", "Ms", " Ms", "Mrs", " Mrs", "They", "they", " They", " they", "A", "He", "She", " ", "", "I", "My", "the", "The"]
    for text_frag in all_text_frags:
        t_temp = []
        for frag in text_frag:
            if frag in bad_guesses:
                continue
            t_temp.append(frag)
        temp.append(t_temp)
    all_text_frags = temp

    def format_prompt(input):
        conv = get_conv_template('llama-3')
        conv.set_system_message(
            f"Please identify words that can be categorized as '{entity}' from the given text. Format the output as list with no additional text. Example: ['{entity}_1', '{entity}_2']." if entity!='Money' else f"Please identify Currency Value from the given text. Format the output as list with no additional text. Example: ['Currency Value 1', 'Currency Value 2']."
        )
        conv.append_message(conv.roles[0], input)
        conv.append_message(conv.roles[1], None)
        text = conv.get_prompt()
        return text

    def collate_fn(batch):
        tokenized_text = tokenizer(batch,  
                                   return_tensors='pt', 
                                   add_special_tokens=False)
        return tokenized_text

    model.eval()
    max_new_tokens=4096

    for all_text_frag in all_text_frags:
        frag_output_dict = {}
        frag_output_dict[entity] = []

        prompts = [format_prompt(text) for text in all_text_frag]
        prompts = any2en(prompts)
        prompt_dataloader = DataLoader(prompts, collate_fn=collate_fn, batch_size=batch_size)

        for i, prompt_batch in tqdm(enumerate(prompt_dataloader), total=len(prompt_dataloader)):
            try:
                with torch.no_grad():
                    prompt_batch.to(device)
                    outputs = model.generate(**prompt_batch, 
                                            do_sample=False,
                                            max_length=max_new_tokens)
                    outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                    outputs = [output.replace("['", "") for output in outputs]
                    outputs = [output.replace("']", "") for output in outputs]
                    outputs = [r.split('assistant\n\n')[-1].strip() for r in outputs]
                    frag_output_dict = postprocess_output(outputs, frag_output_dict, entity)
            except:
                    traceback.print_exc()
                    print(f"Error on batch ID: {i}")

        temp = []
        for kk in frag_output_dict[entity]:
            for kki in kk:
                if kki.strip().replace("'","") not in bad_guesses:
                    temp.append(kki.strip().replace("'",""))
        output_dict[entity].append(temp)
        
    return output_dict

def extract_entities_LLM_sp(
    all_text,
    model,
    tokenizer,
    batch_size,
    device,
    entity_type,
):
    # Prepare input
    max_new_tokens = 4096
    output_dict = {}
    output_dict[entity_type] = []
    all_text_frags = [text.split('.') for text in all_text]
    temp = []
    bad_guesses = ["Mr"," Mr", "mr", "Ms", " Ms", "ms", "prof", "Prof", "Mrs", " Mrs", "They", 
                    "they", " They", " they", "A", "He", "She", " ", "", "I", "My", "the", "The",]
    for text_frag in all_text_frags:
        t_temp = []
        for frag in text_frag:
            if frag in bad_guesses:
                continue
            t_temp.append(frag)
        temp.append(t_temp)
    all_text_frags = temp

    for all_text_frag in all_text_frags:
        frag_output_dict = {}
        frag_output_dict[entity_type] = []

        examples = [
            {
                "conversations": [
                    {"from": "human", "value": f"Text: {text}"}, 
                    {"from": "gpt", "value": "I've read this text."}, 
                    {"from": "human", "value": f"What describes {entity_type} in the text?"}, 
                    {"from": "gpt", "value": "[]"}
                ]
            } for text in all_text_frag # all_text
        ]

        # Batch process helper
        def collate_fn(batch):
                tokenized_text = tokenizer(batch, return_tensors='pt')
                return tokenized_text

        # Preprocess prompts...
        prompts = [preprocess_instance(example['conversations']) for example in examples]
        # print(prompts)
        prompts = any2en(prompts)
        prompt_dataloader = DataLoader(prompts, collate_fn=collate_fn, batch_size=batch_size)
        for i, prompt_batch in tqdm(enumerate(prompt_dataloader), total=len(prompt_dataloader)):
            try:
                with torch.no_grad():
                    prompt_batch.to(device)
                    outputs = model.generate(**prompt_batch, 
                                            do_sample=False,
                                            max_length=max_new_tokens)
                    outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                    outputs = [r.split('ASSISTANT:')[-1].strip() for r in outputs]
                    frag_output_dict = postprocess_output(outputs, frag_output_dict, entity_type)

            except:
                    traceback.print_exc()
                    print(f"Error on batch ID: {i}")
                    input()

        temp = []
        for kk in frag_output_dict[entity_type]:
            for kki in kk:
                if kki.strip() not in bad_guesses:
                    temp.append(kki.strip())
        output_dict[entity_type].append(temp)

    return output_dict

def extract_entities_LLM_sp_old(
    all_text,
    model,
    tokenizer,
    batch_size,
    device,
    entity_type,
):

    # Prepare input
    max_new_tokens = 4096
    output_dict = {}
    output_dict[entity_type] = []

    examples = [
        {
            "conversations": [
                {"from": "human", "value": f"Text: {text}"}, 
                {"from": "gpt", "value": "I've read this text."}, 
                {"from": "human", "value": f"What describes {entity_type} in the text?"}, 
                {"from": "gpt", "value": "[]"}
            ]
        } for text in all_text
    ]

    # Batch process helper
    def collate_fn(batch):
            tokenized_text = tokenizer(batch, return_tensors='pt')
            return tokenized_text

    # Preprocess prompts...
    prompts = [preprocess_instance(example['conversations']) for example in examples]
    print(prompts[-5:])
    prompts = any2en(prompts)
    prompt_dataloader = DataLoader(prompts, collate_fn=collate_fn, batch_size=batch_size)
    for i, prompt_batch in tqdm(enumerate(prompt_dataloader), total=len(prompt_dataloader)):
        try:
            with torch.no_grad():
                prompt_batch.to(device)
                outputs = model.generate(**prompt_batch, 
                                        do_sample=False,
                                        max_length=max_new_tokens)
                outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                outputs = [r.split('ASSISTANT:')[-1].strip() for r in outputs]
                output_dict = postprocess_output(outputs, output_dict, entity_type)
        except:
                traceback.print_exc()
                print(f"Error on batch ID: {i}")
                input()


    return output_dict


def extract_entities_LLM(
    all_text,
    model_path,
    batch_size,
    device,
    entity_type,
):
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    max_new_tokens = 4096
    output_dict = {}
    output_dict[entity_type] = []
    examples = [
        {
            "conversations": [
                {"from": "human", "value": f"Text: {text}"}, 
                {"from": "gpt", "value": "I've read this text."}, 
                {"from": "human", "value": f"What describes {entity_type} in the text?"}, 
                {"from": "gpt", "value": "[]"}
            ]
        } for text in all_text
    ]

    # Batch process helper
    def collate_fn(batch):
            tokenized_text = tokenizer(batch, return_tensors='pt')
            return tokenized_text

    # Preprocess prompts...
    prompts = [preprocess_instance(example['conversations']) for example in examples]
    prompts = any2en(prompts)
    prompt_dataloader = DataLoader(prompts, collate_fn=collate_fn, batch_size=batch_size)

    for i, prompt_batch in tqdm(enumerate(prompt_dataloader), total=len(prompt_dataloader)):
        try:
            with torch.no_grad():
                prompt_batch.to(device)
                outputs = model.generate(**prompt_batch, 
                                         do_sample=False,
                                         max_length=max_new_tokens)
                outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                outputs = [r.split('ASSISTANT:')[-1].strip() for r in outputs]
                output_dict = postprocess_output(outputs, output_dict, entity_type)
        except:
                traceback.print_exc()
                print(f"Error on batch ID: {i}")

    return output_dict


"""
QA methods
"""
### Gemini QA
def QA_gemini(all_text, api_key, entity_type):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    generation_config = genai.types.GenerationConfig(
        temperature=0.7,
        max_output_tokens=4096,
        top_p=1,
    )

    sleep_time = 600
    def query_llm(text, model="gemini-1.5-pro"):
        model = genai.GenerativeModel(
            model_name = model,
            generation_config=generation_config,
            system_instruction=[
                f"Please answer the question based on the summary. Be as concise as possible."
            ]
        )

        temp_text = text
        # print(text)
        response = model.generate_content(temp_text)
        # print(response)
        text = response.text
        return text

    crash_arrays = []
    loop = asyncio.get_event_loop()
    arr = all_text
    counter = range(len(arr))
    crash_array = []
    a = time.time()
    print(f"Starting API NER extraction. Time: {a}")
    # afk = gemini_cleaner(query_llm(arr))
    responses = []
    for question in arr:
        # time.sleep(5)
        response = query_llm(question)
        responses.append(response.replace('\n',''))
        # input()
    print(repr(responses))
    # input()
    print(f"Task target:{len(arr)} Time taken: {time.time()-a}")
    crash_arrays.append(crash_array)
    print(f"Crash arrays:", crash_arrays)

    return {entity_type: responses}

### GPT-4 QA
def QA_GPT(all_text, api_key, entity_type):
    client = OpenAI(api_key=api_key,)
    async def query_llm(text, model="gpt-4o"):
        response = client.chat.completions.create(model=model,
        messages=[
          {
            "role": "system",
              # Prompt format from Ye et al. https://arxiv.org/pdf/2303.10420.pdf
            "content": f"Please answer the question based on the summary. Be as concise as possible."
          },
          {
            "role": "user",
            "content": text 
          },
        ],
        temperature=0,
        max_tokens=4096,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)
        return response

    async def extract_text(text, idx, crash_array):
        try:
            response = await query_llm(text)
            response = response.choices[0].message.content
            response = response.encode().decode()
            return response
        except Exception as e:
            print(e)
            return response

    crash_arrays = []
    loop = asyncio.get_event_loop()
    arr = all_text
    counter = range(len(arr))
    crash_array = []
    a = time.time()
    print(f"Starting API NER extraction. Time: {a}")
    response = loop.run_until_complete(asyncio.gather(*[extract_text(arr[i], i, crash_array) for i in counter],
                                                      return_exceptions=True))
    print(f"Task target:{len(arr)} Time taken: {time.time()-a}")
    crash_arrays.append(crash_array)
    print(f"Crash arrays:", crash_arrays)

    return {entity_type: response}

### Llama-3 QA
def QA_llama(all_text, model_path, batch_size, entity_type, 
             sys_msg="Please answer the question based on the summary. Be as concise as possible.",
             device='cuda:0'):
    entity = entity_type
    output_dict = {}
    output_dict[entity] = []
    def format_prompt(input):
        conv = get_conv_template('llama-3')
        conv.set_system_message(
            sys_msg
        )
        conv.append_message(conv.roles[0], input)
        conv.append_message(conv.roles[1], None)
        text = conv.get_prompt()
        return text

    def collate_fn(batch):
        tokenized_text = tokenizer(batch,  
                                   return_tensors='pt', 
                                   add_special_tokens=False)
        return tokenized_text


    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path,
                                                  use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path,
                                                 device_map=device, torch_dtype=torch.bfloat16,
                                                 attn_implementation='flash_attention_2')
    model.eval()
    max_new_tokens=4096

    prompts = [format_prompt(text) for text in all_text]
    prompts = any2en(prompts)
    prompt_dataloader = DataLoader(prompts, collate_fn=collate_fn, batch_size=batch_size)

    for i, prompt_batch in tqdm(enumerate(prompt_dataloader), total=len(prompt_dataloader)):
        try:
            with torch.no_grad():
                prompt_batch.to(device)
                outputs = model.generate(**prompt_batch, 
                                         do_sample=False,
                                         max_length=max_new_tokens)
                outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                outputs = [output.replace("['", "") for output in outputs]
                outputs = [output.replace("']", "") for output in outputs]
                outputs = [r.split('assistant\n\n')[-1].strip() for r in outputs]
                output_dict = postprocess_output_QA(outputs, output_dict, entity)
        except:
                traceback.print_exc()
                print(f"Error on batch ID: {i}")

    # print(output_dict)
    return output_dict

"""
Translation methods
"""
# OPUS-MT translation
def translate_OPUS_MT(all_text, model_path, device='cuda:0'):
    print(model_path)
    translated_de_encrypted_en = []
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    temp_model = MarianMTModel.from_pretrained(model_path)
    temp_model.to(device)
    temp_model.eval()
    text = any2en(all_text)
    def collate_fn(batch):
        tokenized_text = tokenizer(batch, 
                                   padding=True, 
                                   return_tensors='pt')
        return tokenized_text

    prompt_dataloader = DataLoader(text, collate_fn = collate_fn, batch_size=1)

    with torch.no_grad():
        for i, batch in tqdm(enumerate(prompt_dataloader), total=len(prompt_dataloader)):
            try:
                batch.to(device)
                outputs = temp_model.generate(**batch, 
                                              do_sample=False,
                                             )
                translated_text = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                translated_de_encrypted_en.extend(translated_text)
            except Exception as e:
                translated_de_encrypted_en.append("Error")

    return translated_de_encrypted_en

# GPT-4 Translation
def translate_GPT(all_text, target_lang, api_key):
    client = OpenAI(api_key=api_key,)
    async def query_llm(text, model="gpt-4o"):
        response = client.chat.completions.create(model=model,
        messages=[
          {
            "role": "system",
              # Prompt format from Ye et al. https://arxiv.org/pdf/2303.10420.pdf
            "content": f"You will be provided with sentences in English, and your task is to translate them into {target_lang}. Do not modify any numerical values."
          },
          {
            "role": "user",
            "content": text 
          },
        ],
        temperature=0,
        max_tokens=4096,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)
        return response

    async def extract_text(text, idx, crash_array):
        try:
            response = await query_llm(text)
            response = response.choices[0].message.content
            response = response.encode().decode()
            return response
        except Exception as e:
            print(e)
            return response

    crash_arrays = []
    loop = asyncio.get_event_loop()
    arr = all_text
    counter = range(len(arr))
    crash_array = []
    a = time.time()
    print(f"Starting API NER extraction. Time: {a}")
    translated_de_encrypted_en = loop.run_until_complete(asyncio.gather(*[extract_text(arr[i], i, crash_array) for i in counter],
                                                      return_exceptions=True))
    print(f"Task target:{len(arr)} Time taken: {time.time()-a}")
    crash_arrays.append(crash_array)
    print(f"Crash arrays:", crash_arrays)

    return translated_de_encrypted_en

# Gemini NER
def translate_gemini(all_text, target_lang, api_key):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    generation_config = genai.types.GenerationConfig(
        temperature=0,
        max_output_tokens=4096,
        top_p=1,
    )

    sleep_time = 600
    def query_llm(text, model="gemini-1.5-pro-001"):
        model = genai.GenerativeModel(
            model_name = model,
            generation_config=generation_config,
            system_instruction=[
                f"You will be provided with sentences in English, and your task is to translate them into {target_lang}. Return them as a list of strings and nothing else. DO NOT ADD '.' TO ANY NUMERICAL VALUE."
            ]
        )

        temp_text = text
        response = model.generate_content(temp_text)
        text = response._result.candidates[0].content.parts[0].text
#         text = response
        return text

    crash_arrays = []
    loop = asyncio.get_event_loop()
    arr = all_text
    counter = range(len(arr))
    crash_array = []
    a = time.time()
    print(f"Starting API NER extraction. Time: {a}")
    # afk = gemini_cleaner(query_llm(arr))
    translated_de_encrypted_en = query_llm(arr)
    translated_de_encrypted_en = translated_de_encrypted_en.replace("```","")
    # translated_de_encrypted_en = translated_de_encrypted_en.replace('"',"'")
    translated_de_encrypted_en = translated_de_encrypted_en.replace('«',"")
    translated_de_encrypted_en = translated_de_encrypted_en.replace('»',"")
    translated_de_encrypted_en = translated_de_encrypted_en.replace("python","")
    translated_de_encrypted_en = ast.literal_eval(translated_de_encrypted_en)
    print(f"Task target:{len(arr)} Time taken: {time.time()-a}")
    crash_arrays.append(crash_array)
    print(f"Crash arrays:", crash_arrays)

    return translated_de_encrypted_en

"""
Sanitation methods
"""

def M_epsilon(x, n_lower, n_upper, epsilon, discretization_size=100):
  # Sample within given range to provide privacy guarantee
  n_upper = int(n_upper)
  n_lower = int(n_lower)
  total_range = n_upper-n_lower
  x = (x-n_lower)*discretization_size/total_range
  p_i = []
  for s in range(discretization_size):
      p_i.append(math.exp(-abs(x-s)*epsilon/2))
  p_i = [val/sum(p_i) for val in p_i]
  noised_output = np.random.choice(range(discretization_size),1,p=p_i)*total_range/discretization_size+n_lower
  return int(noised_output[0])


"""
FPE utility functions
"""
def fpe_sanitize_money(value, cipher_fn):
    return format_align_digits(
      cipher_fn.encrypt(
          str(value).replace("$","").replace(",","").replace(".","").replace(" ","")
        ),
        str(value)
    )

def fpe_desanitize_money(value, cipher_fn):
    return format_align_digits(
        cipher_fn.decrypt(
            str(value).replace("$","").replace(",","").replace(".","").replace(" ","")
        ),
        str(value)
    )

def get_encrypted_values(inputs, entity, N, rho, epsilon, c, nd, use_fpe=False):
    # print("Pre-encryption:", entities)
    # inputs is a list of lists.
    new_entities = []
    names_lookup = []
    # Maps ciphertext names to plaintext names. Replace in line.
    names_mapping = dict()
    if entity=='Name':

        if use_fpe:
            countries = ['US','GB','FR']#'CA','DK','FI','RU','LU']# ,'SA','IN','CN','ES','AT','BR', 'CH', 'EG']
            first_names = []
            last_names = []

            for country in countries: 
                first_names = first_names + nd.get_top_names(n=1000,gender='M',country_alpha2=country)[country]['M']

            for country in countries: 
                last_names = last_names + nd.get_top_names(n=1000,country_alpha2=country,use_first_names=False)[country]

            # Get rid of duplicates
            remove_fns = ['Saba','Lanka','Deblog', 'Donas']
            remove_lns = ['guez','quez','ecour','Mai','ü', 'é', 'Behal']

            first_names = list(set(first_names))
            last_names = list(set(last_names))
            print(len(last_names))

            temp = []

            for i in range(len(first_names)):
                flag = 0
                for rfn in remove_fns:
                    if rfn in first_names[i]: 
                        flag = 1
                        break
                if flag==0:
                    temp.append(first_names[i])

            first_names = temp[:]

            for i in range(len(last_names)):
                flag = 0
                for rfn in remove_lns:
                    if rfn in last_names[i]: 
                        flag = 1
                        break
                if flag==0:
                    temp.append(last_names[i])

            last_names = temp[:]
            print(len(last_names))

            first_names = sorted(list(set(first_names)))
            last_names = sorted(list(set(last_names)))

            # Suppose we have n names. Chuck the last k names in the list and plug in these.
            input_first_names = []
            input_last_names = []
            for i, k_input in enumerate(inputs):
                temp_fnames = []
                temp_lnames = []
                for input in k_input:
                    if " " in input:
                        temp_ab = input.split()
                        a, bs = temp_ab[0], temp_ab[1:]
                        input_first_names.append(a)
                        temp_fnames.append(a)
                        for b in bs:
                            input_last_names.append(b)
                            temp_lnames.append(b)
                    else:
                        input_first_names.append(input)
                        temp_fnames.append(input)

                for tt in temp_fnames:
                    inputs[i].append(tt)
            # Get rid of first names like Jean Baptiste
            temp = []
            for name in first_names:
                if " " not in name:
                    temp.append(name)

            first_names = temp[:]

            temp = []
            for name in last_names:
                if " " not in name:
                    temp.append(name)

            last_names = temp[:]
            print(len(last_names))

            # Get rid of repetitions in the list itself.

            excess_fnames = 0
            excess_lnames = 0
            for name in input_first_names:
                if name in first_names:
                    first_names.remove(name)
                    excess_fnames+=1
            for name in input_last_names:
                if name in last_names:
                    last_names.remove(name)
                    excess_lnames+=1

            first_names = first_names[:1000]
            last_names = last_names[:1000]
            first_names = first_names[:min(-len(set(input_first_names)),-1)] + list(set(input_first_names))
            last_names = last_names[:min(-len(set(input_last_names)),-1)] + list(set(input_last_names))
            assert len(first_names)==1000
            assert len(last_names)==1000


        offset = 3

        for k_input in inputs:
            temp = []
            for input in k_input:
                if use_fpe:
                    if " " in input:
                        # If already first-last name, run and conjoin.
                        t_names = input.split()
                        a = t_names[0]
                        pt_fn_idx = first_names.index(a)
                        pt_fn_idx = "0"*(offset-len(str(pt_fn_idx))) + str(pt_fn_idx)
                        pt_idx = "" + pt_fn_idx

                        for b in t_names[1:]:
                            pt_ln_idx = last_names.index(b)
                            pt_ln_idx = "0"*(offset-len(str(pt_ln_idx))) + str(pt_ln_idx)
                            pt_idx += pt_ln_idx

                        ct_idx = fpe_sanitize_money(pt_idx, c)
                        ct_idxs = [ct_idx[i:i+offset] for i in range(0, len(ct_idx), offset)]

                        ct_name = first_names[int(ct_idxs[0])]
                        for b in ct_idxs[1:]:
                            ct_last_name = last_names[int(b)]
                            ct_name = ct_name + " " + ct_last_name

                        temp.append(ct_name)
                    else:
                        a = input
                        pt_fn_idx = first_names.index(a)
                        pt_fn_idx = "0"*(offset-len(str(pt_fn_idx))) + str(pt_fn_idx)
                        pt_idx = "" + pt_fn_idx + "9"*offset

                        ct_idx = fpe_sanitize_money(pt_idx, c)
                        ct_idxs = [ct_idx[i:i+offset] for i in range(0, len(ct_idx), offset)]
                        ct_name = first_names[int(ct_idxs[0])]
                        for b in ct_idxs[1:]:
                            ct_last_name = last_names[int(b)]
                            ct_name = ct_name + " " + ct_last_name

                        temp.append(ct_name)

                    # Grad mapping for decoding and for sanity.
                    names_mapping[ct_name] = input

                else:
                    for k in range(len(input)):
                        temp_name = names.get_full_name(gender='male')
                        temp.append(temp_name)

            new_entities.append(temp)
            # print(new_entities)
        names_lookup.append(first_names)
        names_lookup.append(last_names)

    # Add noise to age and encrypt
    if entity=='Age':
        for input in inputs:
            temp = []
            for real_age in input:
                temp_age = M_epsilon(int(real_age),10,99,epsilon)
                age = str(temp_age)
                temp.append(age)

            new_entities.append(temp)

    # Account for currency symbols. UniNER returns '$' along with the amount.
    if entity=='Money':
        valid_indices = []
        for kk, input in enumerate(inputs):
            temp = []
            trip = 0
            for real_money in input:
                try:
                    if use_fpe:
                        offset = 6
                        val = "9"*offset + str(real_money)
                        money = fpe_sanitize_money(val, c)

                    else:
                        if int(float(real_money))==1:
                            money = round(M_epsilon(int(float(real_money)),1,2,epsilon), 2)
                        elif int(float(real_money))<100:
                            money = round(M_epsilon(int(float(real_money)),2,1000,epsilon), 2)
                        elif int(float(real_money))<1000:
                            money = round(M_epsilon(int(float(real_money)),100,10000,epsilon), 2)
                        else:
                            money = round(M_epsilon(int(float(real_money)),1000,10000,epsilon), 2)

                    temp.append(str(money))
                    names_lookup.append(str(real_money))
                    names_mapping[money] = str(real_money)
                except Exception as e:
                    print(e)
                    trip += 1
                    print("Error value:", real_money)
                    temp.append('None')

            if trip==0: valid_indices.append(kk)
            # print("VALID_INDICES\n", valid_indices)
            new_entities.append(temp)

    return new_entities, names_lookup, names_mapping

def get_decrypted_values(line, entity, tar_extraction, src_extraction, names_lookup, names_mapping, c, use_fpe=False, decrypt_task='translation'):
    if use_fpe and entity=='Money':
        offset=7
        for value in names_mapping:
            val = str(value)
            if len(val)<6: continue
            decrypt = fpe_desanitize_money(val, c)
            decrypt = decrypt[6:]
            if value==None: continue
            if value in line:
                line = line.replace(value, decrypt)
            elif value.replace(".", ",") in line:
                line = line.replace(value.replace(".", ","), decrypt.replace(".", ","))
            else:
                line = line.replace(value.replace(".", ","), decrypt.replace(".", ","))

    elif use_fpe and entity=='Name':
        def check(name_list, target):
            for i, n in enumerate(name_list):
                if finder(name_list[i], target):
                    return i
        def finder(word1, word2):
            encoded1 = unicodedata.normalize('NFC', word1)
            encoded2 = unicodedata.normalize('NFC', word2)
            return encoded1==encoded2

        offset = 3
        first_names, last_names = names_lookup
        if decrypt_task=='long_context':
            decrypt_target = tar_extraction
        else:
            decrypt_target = names_mapping
        for i, name in enumerate(decrypt_target):
            try:
                if " " in name and name not in first_names:
                    t_names = name.split()
                    a = t_names[0]
                    pt_fn_idx = check(first_names, a)

                    pt_fn_idx = "0"*(offset-len(str(pt_fn_idx))) + str(pt_fn_idx)
                    pt_idx = "" + pt_fn_idx

                    for b in t_names[1:]:
                        pt_ln_idx = check(last_names, b)
                        pt_ln_idx = "0"*(offset-len(str(pt_ln_idx))) + str(pt_ln_idx)
                        pt_idx += pt_ln_idx

                    ct_idx = fpe_desanitize_money(pt_idx, c)
                    ct_idxs = [ct_idx[i:i+offset] for i in range(0, len(ct_idx), offset)]

                    ct_name = first_names[int(ct_idxs[0])]
                    for b in ct_idxs[1:]:
                        if b=="9"*offset:
                            continue
                        ct_last_name = last_names[int(b)]
                        ct_name = ct_name + " " + ct_last_name

                    pt_name = ct_name

                else:
                    a = name
                    pt_fn_idx = first_names.index(a)
                    pt_fn_idx = "0"*(offset-len(str(pt_fn_idx))) + str(pt_fn_idx)
                    pt_idx = "" + pt_fn_idx + "9"*offset

                    ct_idx = fpe_desanitize_money(pt_idx, c)
                    ct_idxs = [ct_idx[i:i+offset] for i in range(0, len(ct_idx), offset)]

                    ct_name = first_names[int(ct_idxs[0])]
                    for b in ct_idxs[1:]:
                        ct_last_name = last_names[int(b)]
                        ct_name = ct_name + " " + ct_last_name
            

                    pt_name = ct_name

            except:
                pt_name = "NotFound"
            line = line.replace(name, pt_name)

    else:
        import re
        def replace_word(text, word1, word2):
            pattern = r'\b' + re.escape(word1) + r'\b'
            return re.sub(pattern, word2, text)
        for value in names_mapping:
            line = replace_word(line, value, names_mapping[value])

    return line