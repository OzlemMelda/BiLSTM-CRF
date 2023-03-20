# BiLSTM-CRF
Experiment with three different models: conditional random field (CRF), bidirectional long short-term memory (BiLSTM), and a combination of the two, and their performances on two named entity recognition (NER) datasets.

# File Descriptions
Data/ner:
  The data files for the Named Entity Recognition task.
  It contains the two folders:
  - GMB
    - train
    - dev
    - test
  - wnut16
    - train
    - dev
    - test

Data/src:
  You need to download the pre-trained word embeddings from
  https://nlp.stanford.edu/projects/glove/
  and put them in this folder.

  To download the pre-trained word embeddings, run the following command:
  wget http://nlp.stanford.edu/data/glove.6B.zip

  If it does not work, you can also download the file from the following link:
  https://github.com/allenai/spv2/blob/master/model/glove.6B.100d.txt.gz
  
  Then, unzip the file and put it in the src folder.
  
conlleval.py
  The evaluation script for the Named Entity Recognition task. (Source: https://github.com/sighsmile/conlleval)

main.py:
  The Python code to run experiments with different arguments.

loader.py:
  The Python code for loading the data for the Named Entity Recognition task.

model.py:
  The Python code for the model for the Named Entity Recognition task.

trian.py:
  The Python code for training, testing for the Named Entity Recognition task.

crf.py:
  The Python code for the CRF model for the Named Entity Recognition task.
  
# Example command line to execute an experiment
python main.py --dataset GMB --epoch 50 --use_crf True

# Experiment Results
- I conducted experiments with or without Glove word embeddings and utilized BiLSTM
and BiLSTM + CRF models for NER, comparing their performance to the CRF model.

- The objective is to design and implement a model for named-entity recognition (NER) task that
can achieve high performance on two provided datasets - GMB and WNUT-16. 

- The pre-trained Glove embeddings are utilized to convert the natural language data into
a numerical representation. I train and compare the performance of the
models with and without pre-trained Glove embedding on the two datasets. 

- The evaluation metric for the model is the F1 score.

- Pre-trained embeddings definitely help the model understand the words and improve the performance as can be seen from F1
scores. When we used pre-trained embeddings, the performance of the
model increased for both wnut16 and GMB dataset. We also see performance increase after we add CRF to BiLSTM model. It’s because CRF is
used to improve the result of BiLSTM by imposing adjacency constraints
on neighboring elements in the sequence.

![image](https://user-images.githubusercontent.com/53811688/226488864-ce2d9c50-79b4-4bd8-97bc-fc2a7c878ecc.png)

![image](https://user-images.githubusercontent.com/53811688/226488905-aacec458-c405-417e-a1f5-d49e45e91c6e.png)

*w/o means without pre-trained embedding

- Explanation of WNUT-16 having low F1 score would be this: 
  - Inspecting the counts of each entity tag and the unique,
  top-frequency words, I conclude that many entity tags have many unique
  words in wnut16. 
  - It means that the terms have more variety. 
  - This makes model training harder and affects model performance in a bad way since
  we may not have enough instances to learn the entity tags.
  
  WNUT-16:
  ![image](https://user-images.githubusercontent.com/53811688/226489312-b11a4f45-e64c-4a76-a28c-58f775cd538d.png)
