## Bert Fine-tuning

### Data
Train set - trainSet.csv with 606,823 search terms examples and 1419 categories. The file contains the two columns, the search term and the search term category.
Test set - candidateTestSet.txt with 67,424 search terms examples.

### Install
Install Huggingface’s transformers library and pytorch. Execute the code below to install the library. 
```
!pip install transformers
!pip install torch
```
### Problem Description
For this multi-class classification problem, I decided to use a pre-trained BERT model. I based my choice on:
- BERT is based on the Transformer architecture.
- The vector BERT assigns to a word is a function of the entire sentence, therefore, a word can have different vectors based on the contexts. I think it's very useful to look at both sides to get the right context for such short sequences as search terms.
- BERT splits unknown words into sub-tokens until it finds a known unigrams. This will help to work with possible OOV words.
- BERT can be fine-tuned by adding just a couple of additional output layers.

I will build and compare the performance of 3 different architectures of BERT.

### 1. Model

#### 1.1 bert_version_1
Approach - freeze all the layers of BERT during fine-tuning and append a dense layer and a softmax layer to the architecture.

I decided to use the BERT-large that has 345 million parameters and train the model for 100 epochs. There is a class imbalance in our dataset. So, I will first compute class weights for the labels in the train set and then pass these weights to the loss function so that it takes care of the class imbalance.

* Model - AutoModel('bert-large-uncased')
* Tokenizer - BertTokenizerFast('bert-large-uncased')
* Learning rate: 1e-5
* Loss function: nn.NLLLoss()
* Softmax activation function: nn.LogSoftmax()

#### 1.2 bert_version_2
I took the BertForSequenceClassification class which is designed for classification tasks.

* Model - BertForSequenceClassification ('bert-base-uncased')
* Tokenizer - BertTokenizer ('bert-base-uncased')
* Learning rate: 5e-5

#### 1.3 bert_version_3
In this version, I created a BertClassifier class with a BERT model to extract the last hidden layer of the [CLS] token and a single-hidden-layer feed-forward neural network as the classifier.

* Model: BertModel 
* Tokenizer: BertTokenizer ('bert-base-uncased')
* Loss function: nn.CrossEntropyLoss()
* Learning rate: 5e-5

### 2. Preprocessing and Tokenization Step
To apply the pre-trained BERT, we must use the tokenizer provided by the library. This is because the model has a specific, fixed vocabulary and the BERT tokenizer has a particular way of handling out-of-vocabulary words.

In a preprocessing step, we should:
* add special tokens to the start and end of each sequence ([CLS] and [SEP]), 
* pad and truncate all sequence to a single constant length,
* explicitly specify what are padding tokens with the "attention mask".

BERT works with fixed-length sequences. To choose the max length, I took the length of the longest search term from the training set - 12 tokens.
No more preprocessing was done because the training data has no special characters or punctuation marks.

### Optimizer & Learning Rate Scheduler
The BERT authors have some recommendations for fine-tuning:
- Batch size: 16, 32
- Learning rate: 5e-5, 3e-5, 2e-5
- Number of epochs: 2, 3, 4

I took:
- Batch size: 32
- Learning rate: 5e-5
- Number of epochs is different for each version.

### 3. Training
Train our Bert Classifier for 3 epochs. In each epoch, we will train our model and evaluate its performance on the validation set.

**Training:**

* Unpack our data from the dataloader and load the data onto the GPU.
* Zero out gradients calculated in the previous pass.
* Perform a forward pass to compute logits and loss.
* Perform a backward pass to compute gradients (loss.backward()).
* Clip the norm of the gradients to 1.0 to prevent gradients of being exceptionally small or big. It helps to promote generalization on our dataset.
* Update the model's parameters (optimizer.step()).
* Update the learning rate (scheduler.step()).
* Compute training loss. This alows to look at training and stop if reached some level of consistency.

**Evaluation:**

* Unpack our data and load it onto the GPU.
* Forward pass.
* Compute validation loss to know if the model is overtraining. When training loss decreases, but validation loss goes up - overtraining.
* Compute weighted F-1 score.


### 3. Training Results

#### 3.1 bert_version_1
train/test split - 0.9/0.1
Total Epochs - 100

Epoch 1:
* Training Loss: 7.252
* Validation Loss: 7.235

Epoch 100:
* Training Loss: 5.741
* Validation Loss: 5.586

#### 3.2 bert_version_2
1. train/test split - 0.9/0.1
Total Epochs - 3

Epoch 1:
* Training Loss:       3.376
* Validation Loss:     2.131
* F1 Score (Weighted): 0.534

Epoch 2:
* Training Loss:     1.818
* Validation Loss:     1.773
* F1 Score (Weighted): 0.594

Epoch 3:
* Training Loss:       1.374
* Validation Loss:     1.670
* F1 Score (Weighted): 0.613

#### 3.3 bert_version_3
train/test split - 0.9/0.1
Total Epochs - 2

Epoch 1:
* Training Loss:       3.062
* Validation Loss:     1.981
* Validation Acc:      57.66

Epoch 2:
* Training Loss:       1.647
* Validation Loss:     1.709
* Validation Acc:      62.01  

2. Train on the entire training data
Total Epochs - 4

Epoch 1:
* Training Loss:       2.008

Epoch 2:
* Training Loss:       1.557

Epoch 3:
* Training Loss:       1.143

Epoch 4:
* Training Loss:       0.90

### 5. Performance Metrics
I proceeded with the bert_version_2 model to get the predictions for our training data. 
I used a weighted F-1 score because we have an imbalanced distribution of classes. 
Also, I measured accuracy per class to calculate the accuracy of our predictions vs true labels.

### Model Complexity
Increasing the batch size reduces the training time significantly.
To save on memory during training and boost the training speed, I created an iterator for our dataset using the torch DataLoader class. 


### Weaknesses and Further Improvement
As mentioned before, I worked with the bert_version_2 model to get predictions. 

**Weaknesses**


**Improvements**
* Train on a larger training set. As shown in the bert_version_3 results, training on entire train set gave smaller training loss.
* Increase number of epochs. As the model showed no overfitting, increasing number of epochs can possibly give better results.
* Change learning rate from 5e-5 to smaller. Try
* Change batch size (32) to smaller (16) to get better accuracy.
* Extract Part-of-Speech embeddings and use them in an embedding layer.
* Use other versions of BERT (RoBERTa, which is a compact and faster version of BERT).

