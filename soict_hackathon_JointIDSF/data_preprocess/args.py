import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--text_column", default="test_norm", type=str, help="The name of the text column")
parser.add_argument("--data_path", default="quocanh34/soict_test_dataset", type=str, help="The path to the data")
parser.add_argument("--id_column", default="id", type=str, help="The name of the id column")
parser.add_argument("--split_name", default="test", type=str, help="The name of the split")
parser.add_argument("--output_column", default="sentence", type=str, help="The name of the output column")

parser.add_argument("--data_raw_path", default="raw_data/train_final_20230919.jsonl")
parser.add_argument("--data_processed_path", default="data")

parser.add_argument("--seed", default=42, type=int)
args = parser.parse_args()