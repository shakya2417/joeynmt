from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from pathlib import Path
import os

lang1 = "hi"
lang2 = "en"
lang = lang1+lang2

tatoeba_kwargs = {
  "path": "tatoeba",
  "lang1": lang1,
  "lang2": lang2,
  "date" : "v2021-07-22",
  "ignore_verifications": True,
  # "cache_dir": "test/data/.cache/huggingface"
}
tatoeba_dev = load_dataset(split="train[:1000]", **tatoeba_kwargs)
tatoeba_test = load_dataset(split="train[1000:2000]", **tatoeba_kwargs)
tatoeba_train = load_dataset(split="train[2000:]", **tatoeba_kwargs)

tatoeba_dev, tatoeba_test, tatoeba_train

print(tatoeba_dev['translation'][:3])
print(tatoeba_test['translation'][:3])
print(tatoeba_train['translation'][:3])

dataset_dict = DatasetDict({
  "train": tatoeba_train,
  "validation": tatoeba_dev,
  "test": tatoeba_test
})

os.chdir('../')
data_dir = "test/data/tatoeba_"+lang
dataset_dict.save_to_disk(data_dir)

# Create the config
config = """
name: "tatoeba_{lang}_sp"
joeynmt_version: "2.0.0"

data:
    train: "{data_dir}/train"
    dev: "{data_dir}/validation"
    test: "{data_dir}/test"
    dataset_type: "huggingface"
    #dataset_cfg:           # not necessary for manually saved pyarray daraset
    #    name: "{lang1}-{lang2}"
    sample_dev_subset: 200
    src:
        lang: "{lang1}"
        max_length: 100
        lowercase: False
        normalize: False
        level: "bpe"
        voc_limit: 8240
        voc_min_freq: 1
        voc_file: "{data_dir}/vocab.txt"
        tokenizer_type: "sentencepiece"
        tokenizer_cfg:
            model_file: "{data_dir}/sp.model"

    trg:
        lang: "{lang2}"
        max_length: 100
        lowercase: False
        normalize: False
        level: "bpe"
        voc_limit: 8240
        voc_min_freq: 1
        voc_file: "{data_dir}/vocab.txt"
        tokenizer_type: "sentencepiece"
        tokenizer_cfg:
            model_file: "{data_dir}/sp.model"

""".format(data_dir=data_dir,lang1=lang1,lang2=lang2,lang=lang)
with (Path(data_dir) / "config.yaml").open('w') as f:
    f.write(config)

#! wget https: // raw.githubusercontent.com / joeynmt / joeynmt / main / scripts / build_vocab.py

#!python build_vocab.py {data_dir}/config.yaml --joint

model_dir = "test/data/models/tatoeba_"+lang
config += """
testing:
    n_best: 1
    beam_size: 5
    beam_alpha: 1.0
    batch_size: 256
    batch_type: "token"
    max_output_length: 100
    eval_metrics: ["bleu"]
    #return_prob: "hyp"
    #return_attention: False
    sacrebleu_cfg:
        tokenize: "13a"

training:
    #load_model: "{model_dir}/latest.ckpt"
    #reset_best_ckpt: False
    #reset_scheduler: False
    #reset_optimizer: False
    #reset_iter_state: False
    random_seed: 42
    optimizer: "adam"
    normalization: "tokens"
    adam_betas: [0.9, 0.999]
    scheduling: "warmupinversesquareroot"
    learning_rate_warmup: 2000
    learning_rate: 0.0002
    learning_rate_min: 0.00000001
    weight_decay: 0.0
    label_smoothing: 0.1
    loss: "crossentropy"
    batch_size: 512
    batch_type: "token"
    batch_multiplier: 4
    early_stopping_metric: "bleu"
    epochs: 10
    updates: 20000
    validation_freq: 1000
    logging_freq: 100
    model_dir: "{model_dir}"
    overwrite: True
    shuffle: True
    use_cuda: True
    print_valid_sents: [0, 1, 2, 3]
    keep_best_ckpts: 3

model:
    initializer: "xavier"
    bias_initializer: "zeros"
    init_gain: 1.0
    embed_initializer: "xavier"
    embed_init_gain: 1.0
    tied_embeddings: True
    tied_softmax: True
    encoder:
        type: "transformer"
        num_layers: 6
        num_heads: 4
        embeddings:
            embedding_dim: 256
            scale: True
            dropout: 0.0
        # typically ff_size = 4 x hidden_size
        hidden_size: 256
        ff_size: 1024
        dropout: 0.1
        layer_norm: "pre"
    decoder:
        type: "transformer"
        num_layers: 6
        num_heads: 8
        embeddings:
            embedding_dim: 256
            scale: True
            dropout: 0.0
        # typically ff_size = 4 x hidden_size
        hidden_size: 256
        ff_size: 1024
        dropout: 0.1
        layer_norm: "pre"

""".format(model_dir=model_dir)
with (Path(data_dir) / "config.yaml").open('w') as f:
    f.write(config)   