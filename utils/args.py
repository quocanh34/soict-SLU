import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default= "quocanh34/soict_test_dataset", help='Path to the Hugging Face dataset')
parser.add_argument('--model_path', type=str, help='Path to the wav2vec model')
parser.add_argument('--norm_path', type=str, help='Path to spoken norm model')
parser.add_argument('--token', type=str, help='Token to push to Hugging face')
parser.add_argument('--hgf_infer_result_path', type=str, help='Path to the online Hugging Face dataset after inference')
parser.add_argument('--local_infer_result_path', type=str, help='Path to the local Hugging Face dataset after inference')
parser.add_argument('--num_proc', type=str, default=4, help='Number of processors')
parser.add_argument('--split', type=str, default="train", help='Data split to load')
parser.add_argument('--revision', type=str, help='revision')
args = parser.parse_args()