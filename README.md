# NLP_imdb_review_classification

### Sentiment Analysis with BERT
For the project I will use the IMDB review data and will perform the following to train a sentiment analysis model.

### IMDB Dataset
IMDB Reviews Dataset is a large movie review dataset. The IMDB Reviews dataset is used for binary sentiment classification, whether a review is positive or negative. It contains 25,000 movie reviews for training and 25,000 for testing. All these 50,000 reviews are labeled data that may be used for supervised deep learning. Besides, there is an additional 50,000 unlabeled reviews.

### Process:
1. Transformers - Text Preprocessing
2. Load BERT Classifier and Tokenizer along with Input modules
3. Download the IMDB Reviews Data
4. Create a processed dataset
5. Configure the Loaded BERT model and Train for Fine-tuning
6. Make Predictions with the Fine-tuned Model

### What is BERT?
BERT stands for Bidirectional Encoder Representations from Transformers and it is a state-of-the-art machine learning model used for NLP tasks. Jacob Devlin and his colleagues developed BERT at Google in 2018. Devlin and his colleagues trained the BERT on English Wikipedia (2,500M words) and BooksCorpus (800M words) and achieved the best accuracies for some of the NLP tasks in 2018. There are two pre-trained general BERT variations: The base model is a 12-layer, 768-hidden, 12-heads, 110M parameter neural network architecture, whereas the large model is a 24-layer, 1024-hidden, 16-heads, 340M parameter neural network architecture. Figure 2 shows the visualization of the BERT network created by Devlin et al.

![image](https://user-images.githubusercontent.com/50436546/203155249-ba4ca4d1-be9b-427f-9711-e5359cd41617.png)
![image](https://user-images.githubusercontent.com/50436546/203155268-a9f1eb3b-2292-4d4d-93c3-388927e95751.png)

### Installing Transformers
Installing the Transformers library is fairly easy. Just run the following pip line on a Google Colab cell:

```python
!pip install transformers
```

After the installation is completed, we will load the pre-trained BERT Tokenizer and Sequence Classifier as well as InputExample and InputFeatures. Then, we will build our model with the Sequence Classifier and our tokenizer with BERT’s Tokenizer.

### First step - Text Pre-processing
#### Tokenization: break down each document into single word tokens.
When to use: when the unit of analysis for task is words.

When not to use: when the unit of analysis is not words, but perhaps characters or sentences.

![image](https://user-images.githubusercontent.com/50436546/203155562-c1bffd58-1b44-450c-bf88-7264f38810de.png)

```python
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures

model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
```

### Initial Imports
We will first have two imports: TensorFlow and Pandas.

```python
import tensorflow as tf
import pandas as pd
```

### Get the Data from the Stanford Repo
Then, we can download the dataset from Stanford’s relevant directory with tf.keras.utils.get_file function, as shown below:

```python
URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file(fname="aclImdb_v1.tar.gz", 
                                  origin=URL,
                                  untar=True,
                                  cache_dir='.',
                                  cache_subdir='')
```

### Remove Unlabeled Reviews
To remove the unlabeled reviews, we need the following operations. The comments below explain each operation:

```python
# The shutil module offers a number of high-level 
# operations on files and collections of files.

import os
import shutil

# Create main directory path ("/aclImdb")
main_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
# Create sub directory path ("/aclImdb/train")
train_dir = os.path.join(main_dir, 'train')
# Remove unsup folder since this is a supervised learning task
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)
# View the final train folder

print(os.listdir(train_dir))
```

### Train and Test Split
Now that we have our data cleaned and prepared, we can create text_dataset_from_directory with the following lines. I want to process the entire data in a single batch. That’s why I selected a very large batch size:

```python
# We create a training dataset and a validation 
# dataset from our "aclImdb/train" directory with a 80/20 split.
train = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', batch_size=30000, validation_split=0.2, 
    subset='training', seed=123)
test = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', batch_size=30000, validation_split=0.2, 
    subset='validation', seed=123)
```

![image](https://user-images.githubusercontent.com/50436546/203156189-0ebdb2f9-9f27-4627-9706-6a8c6feb19f5.png)

As you can see, we have 20,000 reviews for training and 5,000 for validation.

### Convert to Pandas to View and Process
Now we have our basic train and test datasets, I want to prepare them for our BERT model. To make it more comprehensible, I will create a pandas dataframe from our TensorFlow dataset object. The following code converts our train Dataset object to train pandas dataframe:

```python
for i in train.take(1):
  train_feat = i[0].numpy()
  train_lab = i[1].numpy()

train = pd.DataFrame([train_feat, train_lab]).T
train.columns = ['DATA_COLUMN', 'LABEL_COLUMN']
train['DATA_COLUMN'] = train['DATA_COLUMN'].str.decode("utf-8")
```

### Here is the first 5 row of our dataset:

```python
train.head()
```

![image](https://user-images.githubusercontent.com/50436546/203156323-0197b243-e1f7-48c5-96a2-fb4a282f3319.png)

I will do the same operations for the test dataset with the following lines:

```python
for j in test.take(1):
  test_feat = j[0].numpy()
  test_lab = j[1].numpy()

test = pd.DataFrame([test_feat, test_lab]).T
test.columns = ['DATA_COLUMN', 'LABEL_COLUMN']
test['DATA_COLUMN'] = test['DATA_COLUMN'].str.decode("utf-8")
```

### First 5 rows of test dataset:

```python
test.head()
```
![image](https://user-images.githubusercontent.com/50436546/203156414-cc283774-b227-48a2-ab8e-5561dedd3988.png)

### Creating Input Sequences
We have two pandas Dataframe objects waiting for us to convert them into suitable objects for the BERT model. We will take advantage of the InputExample function that helps us to create sequences from our dataset. The InputExample function can be called as follows:

```python
InputExample(guid=None,
             text_a = "Hello, world",
             text_b = None,
             label = 1)
```

### Now we will create two main functions:
1 — convert_data_to_examples: This will accept our train and test datasets and convert each row into an InputExample object.

2 — convert_examples_to_tf_dataset: This function will tokenize the InputExample objects, then create the required input format with the tokenized objects, finally, create an input dataset that we can feed to the model.

```python
def convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN): 
  train_InputExamples = train.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[DATA_COLUMN], 
                                                          text_b = None,
                                                          label = x[LABEL_COLUMN]), axis = 1)

  validation_InputExamples = test.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[DATA_COLUMN], 
                                                          text_b = None,
                                                          label = x[LABEL_COLUMN]), axis = 1)
  
  return train_InputExamples, validation_InputExamples

  train_InputExamples, validation_InputExamples = convert_data_to_examples(train, 
                                                                           test, 
                                                                           'DATA_COLUMN', 
                                                                           'LABEL_COLUMN')
  
def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = [] # -> will hold InputFeatures to be converted later

    for e in examples:
        # Documentation is really strong for this method, so please take a look at it
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length, # truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True, # pads to the right by default # CHECK THIS for pad_to_max_length
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
            input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )


DATA_COLUMN = 'DATA_COLUMN'
LABEL_COLUMN = 'LABEL_COLUMN'
```

### We can call the functions we created above with the following lines:

```python
train_InputExamples, validation_InputExamples = convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN)

train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
train_data = train_data.shuffle(100).batch(32).repeat(2)

validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
validation_data = validation_data.batch(32)

Our dataset containing processed input sequences are ready to be fed to the model.

### Configuring the BERT model and Fine-tuning
We will use Adam as our optimizer, CategoricalCrossentropy as our loss function, and SparseCategoricalAccuracy as our accuracy metric. Fine-tuning the model for 2 epochs will give us around 88% accuracy, which is great.

```python
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate= 0.00003, epsilon=0.00000001, clipnorm=1.0), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

model.fit(train_data, epochs=2, validation_data=validation_data)
```

![image](https://user-images.githubusercontent.com/50436546/203156665-7ca9dcf6-ffe2-45ac-abae-d06161cf360b.png)

After our training is completed, we can move onto making sentiment predictions.

### Making Predictions
I created a list of two reviews I created. The first one is a positive review, while the second one is clearly negative.

```python
pred_sentences = ['This was an awesome movie. I watch it twice my time watching this beautiful movie if I have known it was this good',
                  'One of the worst movies of all time. I cannot believe I wasted two hours of my life for this movie',
                  'It is easily one of the most tragic, beautifully written and directed shows of all time and for me at least is at the top of the list of my favourite shows it is tied with Breaking Bad.',
                  'Too many monsters.',
                  'Amazing movie! But I will never recommend it to other people']
```

We need to tokenize our reviews with our pre-trained BERT tokenizer. We will then feed these tokenized sequences to our model and run a final softmax layer to get the predictions. We can then use the argmax function to determine whether our sentiment prediction for the review is positive or negative. Finally, we will print out the results with a simple for loop. The following lines do all of these said operations:


2 — convert_examples_to_tf_dataset: This function will tokenize the InputExample objects, then create the required input format with the tokenized objects, finally, create an input dataset that we can feed to the model.

```python
tf_batch = tokenizer(pred_sentences, max_length=128, padding=True, truncation=True, return_tensors='tf')
tf_outputs = model(tf_batch)
tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
labels = ['Negative','Positive']
label = tf.argmax(tf_predictions, axis=1)
label = label.numpy()
for i in range(len(pred_sentences)):
  print(pred_sentences[i], ": \n", labels[label[i]])
```
![image](https://user-images.githubusercontent.com/50436546/203156820-eb549e0e-d417-4cf1-9f29-41c658a05dc5.png)



### Result
The model achieved ~88% classification accuracy on the validation set with a transformers network with a pre-trained BERT model on the sentiment analysis of the IMDB reviews dataset. The prediction lines with a more obvious positive or negative tone is captured correct and while the model prediction could not catch some ironic sentences using "but".


### Reference:
1. Canvas: Text Preprocessing and Representation Learning.html
2. https://towardsdatascience.com/sentiment-analysis-in-10-minutes-with-bert-and-hugging-face-294e8a04b671
3. https://keras.io/getting_started/faq/#how-can-i-save-a-keras-model
4. https://towardsdatascience.com/mastering-word-embeddings-in-10-minutes-with-tensorflow-41e25da6aa54
5. https://towardsdatascience.com/mastering-word-embeddings-in-10-minutes-with-imdb-reviews-c345f83e054e
