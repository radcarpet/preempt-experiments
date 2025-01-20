import json
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Translation eval for Preempt")
    parser.add_argument("--file_path",type=str,default=None,help="Name, Money, Age, Zipcode, All")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--pred_category", type=str, default=None)

    args = parser.parse_args()
    return args

def load_data(big_fp):
    with open(big_fp, 'r') as fp:
          data = json.load(fp)
    return data   


if __name__=='__main__':
    args = parse_args()
    data = load_data(args.file_path)
    df = pd.DataFrame()

    df['conversation_hash'] = [i for i in range(len(data))]
    df['predicted_category'] = [args.pred_category for i in range(len(data))]
    df['user_query'] = [data[i]['user_query'] for i in range(len(data))]
    df['target_response'] = [data[i]['target_response'] for i in range(len(data))]
    df['pii_units'] = [data[i]['pii_units'] for i in range(len(data))]
    df['redacted_query'] = [data[i]['redacted_query'] for i in range(len(data))]

    df.to_csv(args.save_path, index=False)