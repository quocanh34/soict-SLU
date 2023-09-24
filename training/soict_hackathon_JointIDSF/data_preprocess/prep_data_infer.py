import py_vncorenlp
py_vncorenlp.download_model(save_dir='./')
from datasets import load_dataset, load_from_disk
import pandas as pd
from utils.args import args

def word_segmentation(example):
    example[args.text_column] = rdrsegmenter.word_segment(example[args.text_column])[0]
    return example

def prepare_files(data):
    df_test = pd.DataFrame(data[:][args.text_column], columns=[args.output_column])

    with open("input.txt", "w") as f:
        for i in range(len(df_test)):
            f.write(df_test[args.output_column][i] + "\n")

    ids = pd.DataFrame(data[:][args.id_column], columns=[args.id_column])
    with open("ids.txt", "w") as f:
        for i in range(len(ids)):
            f.write(ids[args.id_column][i]+ "\n")

if __name__ == "__main__":
    # load data
    test_data = load_from_disk(args.data_path)

    # remove unnecessary columns
    cols_to_remove = test_data[args.split_name].column_names
    cols_to_remove.remove(args.text_column)
    cols_to_remove.remove(args.id_column)
    test_data = test_data.remove_columns(cols_to_remove)

    # word segmentation
    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='./')
    data_segmented = test_data[args.split_name].map(word_segmentation)

    # prepare files
    prepare_files(data_segmented)

    