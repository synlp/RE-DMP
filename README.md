# RE-DMP
The code for our ACL-2022 paper titled "Improving Relation Extraction through Syntax-induced Pre-training with Dependency Masking"


Please contact us at `yhtian@uw.edu` if you have any questions.

## Citation

If you use or extend our work, please cite our paper.

```

```

## Requirements

Our code works with the following environment.
* `python=3.7`
* `pytorch=1.7`

## Downloading BERT and XLNet

In our paper, we use [BERT](https://github.com/google-research/bert) and [XLNet](https://github.com/zihangdai/xlnet) as the encoder.
We follow the [instructions](https://huggingface.co/docs/transformers/converting_tensorflow_models) to convert the TensorFlow checkpoints to the PyTorch version.

**Note**: for XLNet, it is possible that the resulting `config.json` misses the hyper-parameter `n_token`. You can manually add it and set its value to `32000` (which is identical to `vocab_size`).

## Train and Test the model

### Pre-training

Go to the `pre-training` folder for more information about model pre-training.

### Fine-tuning

You can find the command lines to train and test models on a small sample data in `run.sh`.

Here are some important parameters:

* `--do_train`: train the model.
* `--do_test`: test the model.
* `--use_bert`: use BERT as encoder.
* `--use_xlnet`: use XLNet as encoder.
* `--bert_model`: the directory of pre-trained BERT/XLNet model.
* `--model_name`: the name of model to save.

