# KR-Medium

A pretrained Korean-specific BERT model developed by Computational Linguistics Lab at Seoul National University.

It is based on our character-level [KR-BERT](https://github.com/snunlp/KR-BERT) model which utilize WordPiece tokenizer.

<br>

### Vocab, Parameters and Data

|                |                              Mulitlingual BERT<br>(Google) |                KorBERT<br>(ETRI) |                              KoBERT<br>(SKT) |                       KR-BERT character |                   KR-Medium |
| -------------: | ---------------------------------------------: | ---------------------: | ----------------------------------: | -------------------------------------: | -------------------------------------: |
|     vocab size |                                        119,547 |                 30,797 |                               8,002 |                                 16,424 |                                 20,000 |
| parameter size |                                    167,356,416 |            109,973,391 |                          92,186,880 |                             99,265,066 |                             102,015,010 |
|      data size | -<br>(The Wikipedia data<br>for 104 languages) | 23GB<br>4.7B morphemes | -<br>(25M sentences,<br>233M words) | 2.47GB<br>20M sentences,<br>233M words | 12.37GB<br>91M sentences,<br>1.17B words |


<br>

The training data for this model is expanded from those of KR-BERT, texts from Korean Wikipedia, and news articles, by addition of legal texts crawled from the National Law Information Center and [Korean Comments dataset](https://www.kaggle.com/junbumlee/kcbert-pretraining-corpus-korean-news-comments). This data expansion is to collect texts from more various domains than those of KR-BERT. The total data size is about 12.37GB, consisting of 91M and 1.17B words.

The user-generated comment dataset is expected to have similar stylistic properties to the task datasets of NSMC and HSD. Such text includes abbreviations, coinages, emoticons, spacing errors, and typos. Therefore, we added the dataset containing such on-line properties to our existing formal data such as news articles and Wikipedia texts to compose the training data for KR-Medium. Accordingly, KR-Medium reported better results in sentiment analysis than other models, and the performances improved with the model of the more massive, more various training data.

This model’s vocabulary size is 20,000, whose tokens are trained based on the expanded training data using the WordPiece tokenizer.

KR-Medium is trained for 2M steps with the maxlen of 128, training batch size of 64, and learning rate of 1e-4, taking 22 hours to train the model using a Google Cloud TPU v3-8.


### Models

#### TensorFlow

* BERT tokenizer, character-based model ([download](https://drive.google.com/file/d/1OWXGqr2Z2PWD6ST3MsFmcjM8c2mr8PkE/view?usp=sharing))

#### PyTorch

* You can import it from Transformers!

```sh
# pytorch, transformers

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-Medium", do_lower_case=False)

model = AutoModel.from_pretrained("snunlp/KR-Medium")

```


### Requirements

- transformers == 4.0.0
- tensorflow < 2.0


## Downstream tasks

* Movie Review Classification on Naver Sentiment Movie Corpus [(NSMC)](https://github.com/e9t/nsmc)
* Hate Speech Detection [(Moon et al., 2020)](https://github.com/kocohub/korean-hate-speech)


#### tensorflow

* After downloading our pre-trained models, put them in a `models` directory.
* Set the output directory (for fine-tuning)
* Select task name: `NSMC` for Movie Review Classification, and `HATE` for Hate Speech Detection


```sh
# tensorflow
python3 run_classifier.py \
  --task_name={NSMC, HATE} \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --do_lower_case=False\
  --max_seq_length=128 \
  --train_batch_size=128 \
  --learning_rate=5e-05 \
  --num_train_epochs=5.0 \
  --output_dir={output_dir}
```

<br>

### Performances

TensorFlow, test set performances


|       | multilingual BERT | KorBERT<br>character | KR-BERT<br>character<br>WordPiece | KR-Medium |
|:-----:|-------------------:|----------------:|----------------------------:|-----------------------------------------:|
| NSMC (Acc) |  86.82   | 89.81  | 89.74 | 90.29 |
| Hate Speech (F1) | 52.03 | 54.33 | 54.53 | 57.91 |




<br>

## Contacts

nlp.snu@gmail.com


