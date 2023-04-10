import os
import numpy as np
import pandas as pd
from collections import deque
import torch.nn as nn
import torch
# import tensorflow as tf


def extract_axis_1(data, indices):
    res = []
    for i in range(data.shape[0]):
        res.append(data[i, indices[i], :])
    res = torch.stack(res, dim=0).unsqueeze(1)
    return res


def to_pickled_df(data_directory, **kwargs):
    for name, df in kwargs.items():
        df.to_pickle(os.path.join(data_directory, name + '.df'))

def pad_history(itemlist,length,pad_item):
    if len(itemlist)>=length:
        return itemlist[-length:]
    if len(itemlist)<length:
        temp = [pad_item] * (length-len(itemlist))
        itemlist.extend(temp)
        return itemlist


# def extract_axis_1(data, ind):
#     """
#     Get specified elements along the first axis of tensor.
#     :param data: Tensorflow tensor that will be subsetted.
#     :param ind: Indices to take (one for each element along axis 0 of data).
#     :return: Subsetted tensor.
#     """

#     batch_range = tf.range(tf.shape(data)[0])
#     indices = tf.stack([batch_range, ind], axis=1)
#     res = tf.gather_nd(data, indices)

#     return res


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs

def calculate_hit(sorted_list,topk,true_items,rewards,r_click,total_reward,hit_click,ndcg_click,hit_purchase,ndcg_purchase):
    for i in range(len(topk)):
        rec_list = sorted_list[:, -topk[i]:]
        for j in range(len(true_items)):
            if true_items[j] in rec_list[j]:
                rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])
                total_reward[i] += rewards[j]
                if rewards[j] == r_click:
                    hit_click[i] += 1.0
                    ndcg_click[i] += 1.0 / np.log2(rank + 1)
                else:
                    hit_purchase[i] += 1.0
                    ndcg_purchase[i] += 1.0 / np.log2(rank + 1)


# class Memory():
#     def __init__(self):
#         self.buffer = deque()
#
#     def add(self, experience):
#         self.buffer.append(experience)
#
#     def sample(self, batch_size):
#         idx = np.random.choice(np.arange(len(self.buffer)),
#                                size=batch_size,
#                                replace=False)
#         return [self.buffer[ii] for ii in idx]

class NeuProcessEncoder(nn.Module):
    def __init__(self, input_size=64, hidden_size=64, output_size=64, dropout_prob=0.4, device=None):
        super(NeuProcessEncoder, self).__init__()
        self.device = device
        
        # Encoder for item embeddings
        layers = [nn.Linear(input_size, hidden_size),
                torch.nn.Dropout(dropout_prob),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, output_size)]
        self.input_to_hidden = nn.Sequential(*layers)

        # Encoder for latent vector z
        self.z1_dim = input_size # 64
        self.z2_dim = hidden_size # 64
        self.z_dim = output_size # 64
        self.z_to_hidden = nn.Linear(self.z1_dim, self.z2_dim)
        self.hidden_to_mu = nn.Linear(self.z2_dim, self.z_dim)
        self.hidden_to_logsigma = nn.Linear(self.z2_dim, self.z_dim)

    def emb_encode(self, input_tensor):
        hidden = self.input_to_hidden(input_tensor)

        return hidden

    def aggregate(self, input_tensor):
        return torch.mean(input_tensor, dim=-2)

    def z_encode(self, input_tensor):
        hidden = torch.relu(self.z_to_hidden(input_tensor))
        mu = self.hidden_to_mu(hidden)
        log_sigma = self.hidden_to_logsigma(hidden)
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return z, mu, log_sigma
    
    def encoder(self, input_tensor):
        z_ = self.emb_encode(input_tensor)
        z = self.aggregate(z_)
        self.z, mu, log_sigma = self.z_encode(z)
        return self.z, mu, log_sigma

    def forward(self, input_tensor):
        self.z, _, _ = self.encoder(input_tensor)
        return self.z


class MemoryUnit(nn.Module):
    # clusters_k is k keys
    def __init__(self, input_size, output_size, emb_size, clusters_k=10):
        super(MemoryUnit, self).__init__()
        self.clusters_k = clusters_k
        self.input_size = input_size
        self.output_size = output_size
        self.array = nn.Parameter(init.xavier_uniform_(torch.FloatTensor(self.clusters_k, input_size*output_size)))
        self.index = nn.Parameter(init.xavier_uniform_(torch.FloatTensor(self.clusters_k, emb_size)))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, bias_emb):
        """
        bias_emb: [batch_size, 1, emb_size]
        """
        att_scores = torch.matmul(bias_emb, self.index.transpose(-1, -2)) # [batch_size, clusters_k]
        att_scores = self.softmax(att_scores)

        # [batch_size, input_size, output_size]
        para_new = torch.matmul(att_scores, self.array) # [batch_size, input_size*output_size]
        para_new = para_new.view(-1, self.output_size, self.input_size)

        return para_new

    def reg_loss(self, reg_weights=1e-2):
        loss_1 = reg_weights * self.array.norm(2)
        loss_2 = reg_weights * self.index.norm(2)

        return loss_1 + loss_2