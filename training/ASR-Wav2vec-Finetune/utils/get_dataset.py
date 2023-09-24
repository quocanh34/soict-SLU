from datasets import load_dataset
import csv
import zipfile
import os
import argparse
import soundfile as sf
import numpy as np
from joblib import Parallel, delayed
import multiprocessing as mp


def save_wav_file(audio_array, file_name, save_path):
    file_name = file_name + ".wav"
    path = save_path + file_name
    sampling_rate = 16000
    data = np.asarray(audio_array)
    sf.write(path, data, sampling_rate)
    del data
    del path
    del file_name


def generate_csv_from_dataset(zip_data, csv_file_path):
    # Write data to CSV using "|" as delimiter
    with open(csv_file_path, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter="|")
        csv_writer.writerow(["path", "transcript"])  # Write header
        csv_writer.writerows(zip_data)


def generate_zip_id_sentence(ds, id_name, sentence_name, data_path):
    ids = ds[id_name]
    sentences = ds[sentence_name]

    for i in range(len(ids)):
        ids[i] = data_path + ids[i] + ".wav"
    data = zip(ids, sentences)
    return data


def get_dataset(
    dataset_name,
    output_train_path,
    output_eval_path,
    csv_file_path,
    id_name="id",
    sentence_name="sentence_norm",
):
    # Load dataset
    dataset = load_dataset(dataset_name)

    if not os.path.exists(output_train_path):
        os.makedirs(output_train_path)

    if not os.path.exists(output_eval_path):
        os.makedirs(output_eval_path)

    num_jobs = mp.cpu_count()  # Adjust the number of parallel jobs as needed

    Parallel(n_jobs=num_jobs)(
        delayed(save_wav_file)(
            dataset["train"][i]["audio"]["array"],
            dataset["train"][i]["id"],
            output_train_path,
        )
        for i in range(len(dataset["train"]))
    )
    Parallel(n_jobs=num_jobs)(
        delayed(save_wav_file)(
            dataset["test"][i]["audio"]["array"],
            dataset["test"][i]["id"],
            output_eval_path,
        )
        for i in range(len(dataset["test"]))
    )
    # Generate CSV file
    csv_train_file_path = "train.csv"
    csv_train_file_path = os.path.join(csv_file_path, csv_train_file_path)
    csv_test_file_path = "test.csv"
    csv_test_file_path = os.path.join(csv_file_path, csv_test_file_path)

    current_directory = os.getcwd()

    output_train_data = os.path.join(current_directory, output_train_path)

    train_zip_data = generate_zip_id_sentence(
        dataset["train"], id_name, sentence_name, output_train_data
    )
    generate_csv_from_dataset(train_zip_data, csv_train_file_path)

    output_test_data = os.path.join(current_directory, output_eval_path)
    test_zip_data = generate_zip_id_sentence(
        dataset["test"], id_name, sentence_name, output_test_data
    )
    generate_csv_from_dataset(test_zip_data, csv_test_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and extract data from a given url"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="thanhduycao/data_soict_train_synthesis_entity",
        help="Name of the dataset to download",
    )

    parser.add_argument(
        "--output_train_path",
        type=str,
        default="../datasets/Train/",
        help="Path to store the downloaded data",
    )

    parser.add_argument(
        "--output_eval_path",
        type=str,
        default="../datasets/Eval/",
        help="Path to store the downloaded data",
    )

    parser.add_argument(
        "--csv_file_path",
        type=str,
        default="../datasets/",
        help="Path to store the csv file",
    )
    parser.add_argument(
        "--id_name",
        type=str,
        default="id",
        help="Name of the column containing the id",
    )
    parser.add_argument(
        "--sentence_name",
        type=str,
        default="sentence_norm",
        help="Name of the column containing the sentence",
    )
    args = parser.parse_args()

    get_dataset(
        args.dataset_name,
        args.output_train_path,
        args.output_eval_path,
        args.csv_file_path,
        args.id_name,
        args.sentence_name,
    )
