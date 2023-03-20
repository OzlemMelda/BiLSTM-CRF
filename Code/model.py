import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from crf import CRF
from typing import Tuple

from train import *

START_TAG = "<START>"
STOP_TAG = "<STOP>"


class NERModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        tag_to_ix: dict,
        embedding_dim,
        hidden_dim,
        num_laters,
        pre_word_embeds=None,
        use_gpu=False,
        use_crf=False
    ):
        super(NERModel, self).__init__()
        self.use_gpu = use_gpu
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_laters
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.use_crf = use_crf
        self.tagset_size = len(tag_to_ix)
        self.start_tag_id = tag_to_ix[START_TAG]
        self.end_tag_id = tag_to_ix[STOP_TAG]
        
        """
        
        Initilization of our NER model.
        
        """
        
        # Dropout
        self.dropout = nn.Dropout(p = 0.1)
        
        # Embedding layer
        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        if pre_word_embeds is not None:
            self.pre_word_embeds = True
            self.word_embeds.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds))
        else:
            self.pre_word_embeds = False
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size = self.embedding_dim, 
                            hidden_size = self.hidden_dim // 2,
                            num_layers = self.num_layers, 
                            bidirectional=True, 
                            batch_first = True)
        
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)       
        
        # CRF init
        if self.use_crf == True:
            self.crf_model = CRF(tag_to_ix = self.tag_to_ix, 
                                 gpu = self.use_gpu, 
                                 start_tag_id = self.start_tag_id, 
                                 stop_tag_id = self.end_tag_id)

            
    def _get_features(self, sentence: torch.Tensor):
        """

        This is the function to get the features of the sentences from the BiLSTM model.

        Args:
            sentence (torch.Tensor): The input sentence to be processed. The shape of the tensor is (batch_size, seq_len, embedding_size).

        Returns:
            torch.Tensor: The output of the BiLSTM model.
        """
        # print(sentence.size()) # torch.Size([32, 34])
        
        embeds = self.word_embeds(sentence)
        # print(embeds.size()) # torch.Size([34, 32, 100])
  
        lstm_out, _ = self.lstm(embeds)
        # print(lstm_out.size()) # torch.Size([34, 32, 8])
        
        lstm_out = self.dropout(lstm_out)
        
        lstm_feats = self.hidden2tag(lstm_out)
        # print(lstm_feats.size()) # torch.Size([32, 34, 24])
        
        lstm_feats[:, 0, self.start_tag_id] = 0
        lstm_feats[:, 0, self.end_tag_id] = 0
        
        return lstm_feats
        

    def forward(self, sentence: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        """
        This is the function that will be called when the model is being trained.
        The loss for BiLSTM-CRF model is the negative log likelihood of the model.
        The loss for BiLSTM model is the cross entropy loss.

        Args:
            sentence (torch.Tensor): The input sentence to be processed.
            tags (torch.Tensor): The ground truth tags of the input sentence.

        Returns:
            scores (torch.Tensor): The output of the model. It is the loss of the model.
        """
      
        lstm_feats = self._get_features(sentence)
        
        if self.use_crf == True:                    
            score = self.crf_model.forward(lstm_feats, tags)
        else:
            output = F.log_softmax(lstm_feats, dim=2)            
            output = lstm_feats.permute(0, 2, 1)
            score = F.cross_entropy(output, tags)
            
        return score

    
    def inference(self, sentence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This is the function that will be called when the model is being tested.
        
        Args:
            sentence (torch.Tensor): The input sentence to be processed.

        Returns:
            The score and the predicted tags of the input sentence.
            score (torch.Tensor): The score of the predicted tags.
            tag_seq (torch.Tensor): The predicted tags of the input sentence.
            
        """
         # Get the emission scores from the BiLSTM
        lstm_feats = self._get_features(sentence)
        
        if self.use_crf == True:
            score, tag_seq = self.crf_model.inference(lstm_feats)
        else:
            tag_seq = F.log_softmax(lstm_feats, dim=2)
            tag_seq = torch.max(lstm_feats.squeeze(), -1).indices
            score = 0 # dummy for now
            
        return score, tag_seq
        