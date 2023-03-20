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
