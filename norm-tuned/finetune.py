import os
import torch
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from data_handling import DataCollatorForNormSeq2Seq
from model_handling import EncoderDecoderSpokenNorm
from trainer import SpokenNormTrainer
import model_handling
import data_handling
import debug_cross_attention

# load pretrained model
tokenizer = model_handling.init_tokenizer()
data_collator = DataCollatorForNormSeq2Seq(tokenizer)
model = EncoderDecoderSpokenNorm.from_pretrained('nguyenvulebinh/spoken-norm-taggen-v2', cache_dir=model_handling.cache_dir)

# freeze 
for params in model.encoder.embeddings.parameters():
  params.requires_grad = False

# finetune
debug_cross_attention.is_debug = False
if __name__ == "__main__":
    # init model
    model_fake, tokenizer = model_handling.init_model() # train tu dau thi doi thanh model

    # init data
    raw_datasets = data_handling.init_data()
    data_collator = data_handling.DataCollatorForNormSeq2Seq(tokenizer, model=model)

    # set training arguments - these params are not really tuned, feel free to change
    num_epochs = 55
    checkpoint_path = "/oov_checkpoints"
    batch_size = 64 # change to 48 for full training
    training_args = Seq2SeqTrainingArguments(
        output_dir=checkpoint_path,
        learning_rate=8e-06,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        # evaluation_strategy="steps",
        save_strategy="epoch",
        gradient_accumulation_steps=1,
        predict_with_generate=True,
        save_total_limit=5,
        do_train=True,
        do_eval=True,
        logging_steps=2000,  # set to 2000 for full training
        save_steps=500,  # set to 500 for full training
        eval_steps=7500,  # set to 7500 for full training
        warmup_steps=3000,  # set to 3000 for full training
        # max_steps=16, # delete for full training
        num_train_epochs=num_epochs,  # uncomment for full training
        warmup_ratio=1 / num_epochs,
        logging_dir=os.path.join(checkpoint_path, 'log'),
        overwrite_output_dir=True,
        # metric_for_best_model='wer',
        # greater_is_better=False,
        # metric_for_best_model='bleu',
        # greater_is_better=True,
        eval_accumulation_steps=10,
        dataloader_num_workers=4,  # 20 for full training
        generation_max_length=50,
        # sharded_ddp="simple",
        # local_rank=2,
        # fp16=True,
        ignore_data_skip=True
    )

    # instantiate trainer
    trainer = SpokenNormTrainer(
        model=model,
        args=training_args,
        # compute_metrics=metric_handling.get_wer_metric_compute_fn(tokenizer),
        train_dataset=raw_datasets['train'],
        eval_dataset=raw_datasets['test'],
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    trainer.train()
    trainer.model.push_to_hub("linhtran92/finetuned_taggenv2__encoder_embeddings", token="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")
    # trainer.evaluate()
    # trainer.save_model(checkpoint_path)
    # trainer.evaluate()
