from datasets import load_dataset
import utils
import re

chars_to_ignore = r'[,?.!\-;:"“\']/'
patterns_to_replace = {
    r'ki a': 'kia',
}

def replace_patterns(sentence, patterns_to_replace):
    modified_sentence = sentence
    for pattern, replacement in patterns_to_replace.items():
        modified_sentence = re.sub(pattern, replacement, modified_sentence)
    return modified_sentence

def clean_string(s):
    return ' '.join(s.split()).lower()

def norm_sentences(example):
    example["sentence"] = re.sub(chars_to_ignore, '', example["sentence"]).lower()

    sentence = example["sentence"]
    example["sentence_norm_v2"] = utils.norm_word(sentence)
    example["sentence_norm_v2"] = replace_patterns(example["sentence_norm_v2"], patterns_to_replace)
    example["sentence_norm_v2"] = clean_string(example["sentence_norm_v2"])

    # print(example["sentence_norm_v2"])
    return example

if __name__ == "__main__":
    ds = load_dataset("quocanh34/soict_train_dataset")
    ds_norm = ds.map(norm_sentences)
    ds_norm.push_to_hub("thanhduycao/soict_train_dataset", token="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")