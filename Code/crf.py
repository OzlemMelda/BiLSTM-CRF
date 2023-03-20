import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


def logsumexp(log_Tensor: torch.Tensor, axis=-1) -> torch.Tensor:  # shape (batch_size,n,m)
    """
    This is the function to do logsumexp

    Args:
        log_Tensor (torch.Tensor): the input tensor to do logsumexp 
        axis (int, optional): Defaults to -1. the axis to do logsumexp

    Returns:
        torch.Tensor: the result of logsumexp
    """
    return torch.max(log_Tensor, axis)[0]+torch.log(torch.exp(log_Tensor-torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0], -1, 1)).sum(axis))


class CRF(nn.Module):

    """
    This is the class to do CRF.
    """

    def __init__(self, tag_to_ix: Dict, gpu: bool, start_tag_id: int, stop_tag_id: int):
        """
        This is the init function of CRF

        Args:
            tag_to_ix (dict): the dict of tag to index
            gpu (bool): whether use gpu
            start_tag_id (int): the index of start tag
            stop_tag_id (int): the index of stop tag
        """
        super(CRF, self).__init__()
        self.tagset_size = len(tag_to_ix)
        self.device = gpu
        self.start_label_id = start_tag_id
        self.stop_label_id = stop_tag_id
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[self.start_label_id, :] = -10000
        self.transitions.data[:, self.stop_label_id] = -10000
        self.tag_to_ix = tag_to_ix

    def inference(self, feats: torch.Tensor) -> torch.Tensor:
        """

        This is the function to do inference, use it to get the best path of the input feats.

        Args:
            feats (torch.Tensor): the input feats

        Returns:
            torch.Tensor: the best path of the input feats
        """
        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        log_delta = torch.Tensor(
            batch_size, 1, self.tagset_size).fill_(-10000.).to(self.device)
        log_delta[:, 0, self.start_label_id] = 0

        # psi is for the vaule of the last latent that make P(this_latent) maximum.
        psi = torch.zeros((batch_size, T, self.tagset_size), dtype=torch.long).to(
            self.device)  # psi[0]=0000 useless
        for t in range(1, T):
            # delta[t][k]=max_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # delta[t] is the max prob of the path from  z_t-1 to z_t[k]
            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            # psi[t][k]=argmax_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # psi[t][k] is the path choosed from z_t-1 to z_t[k],the value is the z_state(is k) index of z_t-1
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        # trace back
        path = torch.zeros((batch_size, T), dtype=torch.long).to(self.device)

        # max p(z1:t,all_x|theta)
        max_score, path[:, -1] = torch.max(log_delta.squeeze(), -1)

        for t in range(T-2, -1, -1):
            # choose the state of z_t according the state choosed of z_t+1.
            path[:, t] = psi[:, t +
                             1].gather(-1, path[:, t+1].view(-1, 1)).squeeze()

        return max_score, path

    def forward(self, feats: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        """

        This is the forward function of CRF, use it to get the loss of the input feats and tags.

        Args:
            feats (torch.Tensor): the input feats
            tags (torch.Tensor): the input tags
        Returns:
            torch.Tensor: the loss of the input feats and tags
        """

        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # alpha_recursion,forward, alpha(zt)=p(zt,bar_x_1:t)
        log_alpha = torch.Tensor(
            batch_size, 1, self.tagset_size).fill_(-10000.).to(self.device)
        # normal_alpha_0 : alpha[0]=Ot[0]*self.PIs
        # self.start_label has all of the score. it is log,0 is p=1
        log_alpha[:, 0, self.start_label_id] = 0

        # feats: sentances -> word embedding -> lstm -> MLP -> feats
        # feats is the probability of emission, feat.shape=(1,tag_size)
        for t in range(1, T):
            log_alpha = (logsumexp(self.transitions + log_alpha,
                         axis=-1) + feats[:, t]).unsqueeze(1)

        # log_prob of all barX
        forward_score = logsumexp(log_alpha)

        batch_transitions = self.transitions.expand(
            batch_size, self.tagset_size, self.tagset_size)
        batch_transitions = batch_transitions.flatten(1)

        gold_score = torch.zeros((feats.shape[0], 1)).to(self.device)
        # the 0th node is start_label->start_word, the probability of them=1. so t begin with 1.
        for t in range(1, T):
            gold_score = gold_score + \
                batch_transitions.gather(-1, (tags[:, t]*self.tagset_size+tags[:, t-1]).view(-1, 1)) \
                + feats[:, t].gather(-1, tags[:, t].view(-1, 1)).view(-1, 1)

        return torch.mean(forward_score - gold_score)
