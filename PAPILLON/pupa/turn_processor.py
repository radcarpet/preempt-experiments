from create_privacy_span import process_user_query
from filter_context_dependence import context_independence
import json
import argparse
from tqdm import tqdm
import time

"""
1. Get queries [100,500] chars from data for language translation. 
2. Pass to gpt-4o till we have 300 for name, age, money (check if queries are identical)
3. Redacted versions will be obtained with the same. Save somewhere! 
"""

def save_fn(dataset, big_fp):
    with open(big_fp, 'w') as fp:
        json.dump(dataset, fp, indent=2, sort_keys=True)

def load_data(big_fp):
    with open(big_fp, 'r') as fp:
          data = json.load(fp)
    return data 

def process_query_response_pairs(query, response, entity):
    # if not context_independence(query, history):
    #     return None
    pii_units, redacted_query = process_user_query(query, entity)
    return {
        "user_query": query,
        "target_response": response,
        "pii_units": pii_units,
        "redacted_query": redacted_query
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Translation eval for PAPILLON")
    parser.add_argument("--data_path",type=str,default=None,help="Path to finetuning data")
    parser.add_argument("--lang",type=str,default=None,help="Finetuning language")
    parser.add_argument("--entity",type=str,default=None,help="Entity")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    data = load_data(args.data_path)
    processed_queries = []
    entity = args.entity
    # task_line = "Please answer the question based on the summary. Be as concise as possible.\n\n"
    if args.lang=='de':
        task_line = "Please translate this sentence into German: "
    else:
        task_line = "Please translate this sentence into French: "
    print(args)
    print(len(data['queries']))
    # exit()
    for query, response in tqdm(zip(data['queries'], data['responses']), total=100000):
        # print("\nQuery:", query)
        # print("Response:", response)
        processed_query = process_query_response_pairs(query, response, entity)
        # print(processed_query)
        if processed_query['pii_units'] is not None:
            processed_query['user_query'] = task_line + processed_query['user_query']
            processed_query['redacted_query'] = task_line + processed_query['redacted_query']
            processed_queries.append(processed_query)
            print(len(processed_queries))
            if len(processed_queries)%10==0:
                save_fn(processed_queries, f'./datasets/papillon_finetuning_translation_task_processed_{args.lang}_{args.entity}.json')
            if len(processed_queries)==100: break

        time.sleep(2)
    
    save_fn(processed_queries, f'./datasets/papillon_finetuning_translation_task_processed_{args.lang}_{args.entity}.json')