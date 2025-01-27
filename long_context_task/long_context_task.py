import sys
sys.path.insert(0, '../universal-ner')
from src.utils import *
import torch
import evaluate
import numpy as np
import pandas as pd
from pyfpe_ff3 import FF3Cipher
import argparse
import json
import os
import traceback

sys.path.insert(0, '../utils')
from utils_backup import *

"""
# PIPELINE:

1. Use the finetuned NER model (Uni-NER)/other open source models to extract entities.
2. Use appropriate security scheme for encryption
3. Answer with offline/online models
4. Use the finetuned NER model/other open source models to extract entities
5. Use appropriate security scheme for decryption
6. Comparisons
"""

def entity_extractor_ans(model,tokenizer,ner_path,data,batch_size,entity,device,):

    if "UniNER" in ner_path or 'uniner' in ner_path or 'universal' in ner_path:
        ner_name = 'uniner'
    elif 'gemma' in ner_path:
        ner_name = 'gemma'
    elif 'llama' in ner_path or 'Llama' in ner_path:
        ner_name = 'llama'

    entity_extractor = {
        "uniner": extract_entities_LLM_sp_old,
        "gemma": extract_entities_gemma,
        "llama": extract_entities_llama,
    }

    extracted = entity_extractor[ner_name](
        all_text=data,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        entity_type=entity,
        device=device,
    )[entity]

    return extracted

def entity_extractor(model,tokenizer,ner_path,data,batch_size,entity,device,):

    if "UniNER" in ner_path or 'uniner' in ner_path or 'universal' in ner_path:
        ner_name = 'uniner'
    elif 'gemma' in ner_path:
        ner_name = 'gemma'
    elif 'llama' in ner_path:
        ner_name = 'llama'

    entity_extractor = {
        "uniner": extract_entities_LLM_sp,
        "gemma": extract_entities_gemma_sp,
        "llama": extract_entities_llama_sp,
    }

    extracted = entity_extractor[ner_name](
        all_text=data,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        entity_type=entity,
        device=device,
    )[entity]

    return extracted


###################################
##### 0. Data preprocessing #######
###################################

def parse_args():
    parser = argparse.ArgumentParser(description="Translation eval for Preempt")
    parser.add_argument("--entity",type=str,default=None,help="Name, Money, Age, Zipcode, All")
    parser.add_argument("--device",type=str,default='cuda:0',help="Device to mount",)
    parser.add_argument("--qa_path",type=str,default='gpt-4',help="Model for QA. Use path",)
    parser.add_argument("--batch_size",type=int,default=1,help="Batch size",)
    parser.add_argument("--seed",type=int,default=22,help="Crypto seed")
    parser.add_argument("--ner_path",type=str,default=None,help="NER model path",)
    parser.add_argument("--sts_path",type=str,default=None,help="STS model path",)
    parser.add_argument("--tag",type=str,default=None,help="Tag for saving results")
    parser.add_argument("--samples",type=int,default=50,help="Samples for testing")
    parser.add_argument("--start",type=int,default=0,help="Samples for testing")
    parser.add_argument("--end",type=int,default=50,help="Samples for testing")
    parser.add_argument("--exact_indices",type=str,default=None,help="Samples for testing")
    parser.add_argument("--api_key",type=str,default=None,help="API key for online translation")
    parser.add_argument("--use_fpe",type=bool,default=False,help="Use FPE for names and money")
    parser.add_argument("--parse_data",type=bool,default=False,help="Gather data")
    parser.add_argument("--metrics_only",type=bool,default=False,help="Get metrics and update.")
    
    args = parser.parse_args()
    return args

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def save_fn(dataset, big_fp):
    with open(big_fp, 'w') as fp:
        json.dump(dataset, fp, indent=2, sort_keys=True)

def load_data(big_fp):
    with open(big_fp, 'r') as fp:
          data = json.load(fp)
    return data        

args = parse_args()
seed_everything(args.seed)

print(args)

ner_model_path = args.ner_path
qa_model_path = args.qa_path

device = args.device
named_entity = args.entity
batch_size = args.batch_size
api_key = args.api_key
entity_type = args.entity

max_new_tokens = 4096

nd = None   
if args.use_fpe and named_entity=='Name':
    from names_dataset import NameDataset
    nd = NameDataset()

##### Used for FPE.
N = {'Money': 100000} 
rho = {'Money': .05} 
epsilon = 1
key = "EF4359D8D580AA4F7F036D6F04FC6A94"
tweak = "D8E7920AFA330A73" 

cipher_fn = FF3Cipher(key, 
                      tweak,
                      allow_small_domain=True, 
                      radix=10
                     )

if not args.metrics_only:
    all_data = {
        'extraction': {},
        'data_raw': [],
        'data_encrypted': [],
        'data_encrypted_answered': [],
        'answered_data_decrypted': [],
        'data_plain_answered': [],
        'args': vars(args)
    }

    extracted_dict = {
        'src_extraction': {},
        'ans_extraction': {},
        'src_encrypted': {},
        'ans_decrypted': {},
    }

    #### NER extraction
    tokenizer = AutoTokenizer.from_pretrained(ner_model_path)
    model = AutoModelForCausalLM.from_pretrained(ner_model_path, torch_dtype=torch.float16)
    model.to(device)
    model.eval()

    #### Data
    print("Data preprocessing...")
    all_data = load_data('./datasets/long_context_task/all_data.json')

    assert len(all_data['data_raw'])==50
    data = all_data['data_raw'][args.start:args.end][:args.samples]

    ###################################
    ##### 1. NER extraction ###########
    ###################################
    entity=named_entity
    print("\n\nExtraction...\n")
    input_prompts = [f"SUMMARY:\n{data[i]['summary']}\nQUESTION:\n{data[i]['question']}" for i in range(len(data))]
    answers = [data[i]['answers'] for i in range(len(data))]
    extracted = entity_extractor(model,tokenizer,ner_model_path,input_prompts,batch_size,named_entity,device)
    print(extracted[:-5], '\n')
    
    # Save this!
    extracted_dict['src_extraction'][entity] = extracted
    all_data['extraction'] = extracted_dict
    tag = f'{args.tag}/all_data.json'
    os.makedirs(args.tag, exist_ok=True)
    save_fn(all_data, tag)


    ###################################
    ##### 2. Sanitization #############
    ###################################
    print("\n\nSanitization...\n")
    encrypted_values, names_lookup, names_mapping = get_encrypted_values(
            inputs=extracted_dict['src_extraction'][entity],
            entity=entity,
            N=N, rho=rho, epsilon=epsilon, c=cipher_fn, nd=nd, use_fpe=args.use_fpe)

    extracted_dict['src_encrypted'][entity] = encrypted_values

    data_encrypted = []
    for i, line in enumerate(input_prompts):
        # Substitute for all entities
        # Get extracted/encrypted data for the ith line.
        # Substitute all values.
        for value, encrypt in zip(extracted_dict['src_extraction'][entity][i], extracted_dict['src_encrypted'][entity][i]):
            line = line.replace(value, encrypt)

        data_encrypted.append(line)

    # Save this!
    all_data['data_encrypted'] = data_encrypted
    tag = f'{args.tag}/all_data.json'
    save_fn(all_data, tag)

    print(data_encrypted[-5:], '\n')

    ###################################
    ##### 3. QA #######################
    ###################################
    print("\n\nAnswering...\n")
    # Answer with GPT:
    if qa_model_path=='gpt-4':
        data_encrypted_answered = QA_GPT(data_encrypted, api_key, named_entity)
    elif qa_model_path=='gemini':
        data_encrypted_answered = QA_gemini(data_encrypted, api_key, named_entity)
    else:
        data_encrypted_answered = QA_llama(data_encrypted, qa_model_path, batch_size, named_entity)

    # Save this!
    all_data['data_encrypted_answered'] = data_encrypted_answered
    tag = f'{args.tag}/all_data.json'
    save_fn(all_data, tag)

    print(data_encrypted_answered[named_entity][:-5], '\n')


    ###################################
    ##### 4. NER extraction ###########
    ###################################
    print("\n\nExtraction...\n")
    input_prompts = [data_encrypted_answered[entity][i] for i in range(len(data_encrypted_answered[entity]))]
    extracted = entity_extractor_ans(model,tokenizer,ner_model_path,input_prompts,batch_size,named_entity,device)
    print(extracted[:-5], '\n')
    # Save this!
    extracted_dict['ans_extraction'][entity] = extracted
    all_data['extraction'] = extracted_dict
    tag = f'{args.tag}/all_data.json'
    save_fn(all_data, tag)


    ###################################
    ##### 5. Decryption ###############
    ###################################
    print("Decrpytion...")
    # Do entity based substitution. For a list of lists, return a list of lists.
    data_decrypted = []
    # Get the ith data string.
    print(input_prompts)
    errors = []
    for i, line in enumerate(input_prompts):
        # Substitute for all entities
        try:
            # Get extracted/encrypted data for the ith line.
            # Substitute all values.
            line = get_decrypted_values(line,
                                        entity,
                                        extracted_dict['ans_extraction'][entity][i], 
                                        extracted_dict['src_extraction'][entity][i],
                                        names_lookup,
                                        names_mapping,
                                        cipher_fn,
                                        use_fpe=args.use_fpe,
                                        decrypt_task='long_context'
                                        )

            # for value, decrypt in zip(extracted_dict['tar_extraction'][entity][i], extracted_dict['src_extraction'][entity][i]):
            #     line = line.replace(value, decrypt)

            data_decrypted.append(line)
        except Exception as e:
            print("ERROR", i)
            print(traceback.format_exc())
            errors.append(i)

    print(errors)
    # Save this!
    all_data['answered_data_decrypted'] = data_decrypted
    tag = f'{args.tag}/all_data.json'
    save_fn(all_data, tag)

    print(data_decrypted[:-5], '\n')

    ###################################
    ##### 6. Plain QA, for metrics ####
    ###################################
    print("\n\nPlain answering...\n")
    input_prompts = [f"SUMMARY:\n{data[i]['summary']}\nQUESTION:\n{data[i]['question']}" for i in range(len(data))]
    # Answer with model:
    if qa_model_path=='gpt-4':
        data_plain_answered = QA_GPT(input_prompts, api_key, named_entity)
    elif qa_model_path=='gemini':
        data_plain_answered = QA_gemini(input_prompts, api_key, named_entity)
    else:
        data_plain_answered = QA_llama(input_prompts, qa_model_path, batch_size, named_entity)

    # Save this!
    data_plain_answered = [data_plain_answered[entity][i] for i in range(len(data_plain_answered[entity]))]
    all_data['data_plain_answered'] = data_plain_answered
    tag = f'{args.tag}/all_data.json'
    save_fn(all_data, tag)

    print(data_plain_answered[-5:], '\n')
    print("\nNAMES MAPPING:\n", names_mapping)
else:
    all_data = load_data(f'{args.tag}/all_data.json')
    answers = all_data['reference_answers']


###################################
##### 7. Metrics ##################
###################################
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(args.sts_path)
def sts(preds, refs):
    pred_embds = model.encode(preds)
    ref_embds = model.encode(refs)
    similarity = model.similarity(pred_embds, ref_embds)
    avg_similarity = np.mean([similarity[i,i] for i in range(len(similarity))])
    avg_disimilarity = np.mean([(torch.sum(similarity[i,:])-similarity[i,i])/(len(similarity)-1) for i in range(len(similarity))])
    return float(avg_similarity), float(avg_disimilarity)


print("Metrics...")
print("References:")
print(answers[-5:], '\n')
# Score translation:    
metric = evaluate.load('bleu')
predictions = all_data['answered_data_decrypted']
print("\nPreempt answers:")
print(predictions[-5:], '\n')
references = answers
bleu_score = metric.compute(predictions=predictions, references=references)
print(f"\nSanitized BLEU:\n", bleu_score)
all_data['reference_answers'] = answers
all_data['sanitized_bleu_score'] = bleu_score
sts_score, dis_score = sts(predictions, references)
print(f"\nSanitized STS:\n", sts_score)
print(f"\nSanitized Dissimilarity:\n", dis_score)

all_data['sanitized_sts_score'] = sts_score
all_data['sanitized_dis_score'] = dis_score
sts_score, dis_score = sts(all_data['data_encrypted_answered'][entity_type], references)
print(f"\nEncrypted Preempt STS:\n", sts_score)
print(f"\nEncrypted Preempt Dissimilarity:\n", dis_score)
all_data['encrypted_sts_score_wrt_gt'] = sts_score
all_data['encrypted_dis_score_wrt_gt'] = dis_score

metric = evaluate.load('bleu')
predictions = all_data['data_plain_answered']
print("\nPlain answers:")
print(predictions[-5:], '\n')
bleu_score = metric.compute(predictions=predictions, references=references)
print(f"\nPlain:\n", bleu_score)
all_data['plain_bleu_score'] = bleu_score
sts_score, dis_score = sts(predictions, references)
print(f"\nPlain STS:\n", sts_score)
print(f"\nPlain Dissimilarity:\n", dis_score)
all_data['plain_sts_score'] = sts_score
all_data['plain_dis_score'] = dis_score

# Score translation:
metric = evaluate.load('bleu')
predictions = all_data['answered_data_decrypted']
print("\nPreempt answers:")
print(predictions[-5:], '\n')
references = answers
bleu_score = metric.compute(predictions=predictions, references=all_data['data_plain_answered'])
print(f"\nSanitized vs Plain:\n", bleu_score)
all_data['mutual_bleu_score'] = bleu_score
sts_score, dis_score = sts(predictions, all_data['data_plain_answered'])
print(f"\nSanitized vs Plain STS:\n", sts_score)
print(f"\nSanitized vs Plain Dissimilarity:\n", dis_score)
all_data['mutual_sts_score'] = sts_score
all_data['mutual_dis_score'] = dis_score

sts_score, dis_score = sts(all_data['data_encrypted_answered'][entity_type], all_data['data_plain_answered'])
print(f"\nEncrypted Mutual STS:\n", sts_score)
print(f"\nEncrypted Mutual Dissimilarity:\n", dis_score)
all_data['encrypted_sts_score_wrt_plain'] = sts_score
all_data['encrypted_dis_score_wrt_plain'] = dis_score

tag = f'{args.tag}/all_data.json'
save_fn(all_data, tag)
