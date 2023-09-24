import zipfile
from utils import runcmd
import os
import argparse


def get_noise_dataset(url_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    runcmd(f"wget -P {output_path} {url_path}", verbose=True)
    # list to store files
    res = []

    # Iterate directory
    for file_path in os.listdir(output_path):
        # check if current file_path is a file
        if os.path.isfile(os.path.join(output_path, file_path)):
            # add filename to list
            res.append(file_path)

    zip_path = os.path.join(output_path, res[0])
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_path)
    os.remove(zip_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and extract data from a given url"
    )
    parser.add_argument(
        "--url_path",
        type=str,
        default="https://www.openslr.org/resources/28/rirs_noises.zip",
        help="URL path to download data",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../noise_data",
        help="Path to save data",
    )
    args = parser.parse_args()
    get_noise_dataset(args.url_path, args.output_path)
