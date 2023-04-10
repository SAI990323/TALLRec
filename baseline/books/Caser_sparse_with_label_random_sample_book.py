import numpy as np
import pandas as pd
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import os
import logging
import time as Time
from utility import pad_history,calculate_hit
from collections import Counter
import copy
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter   
import random
from sklearn.metrics import roc_auc_score

writer = SummaryWriter('./path/to/log')

logging.getLogger().setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='yc',
                        help='yc, ks, rr')
    # parser.add_argument('--pretrain', type=int, default=1,
    #                     help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--num_filters', type=int, default=16,
                        help='num_filters')
    parser.add_argument('--filter_sizes', nargs='?', default='[2,3,4]',
                        help='Specify the filter_size')
    parser.add_argument('--r_click', type=float, default=0.2,
                        help='reward for the click behavior.')
    parser.add_argument('--r_buy', type=float, default=1.0,
                        help='reward for the purchase behavior.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--model_name', type=str, default='Caser_bce',
                        help='model name.')
    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')
    parser.add_argument('--cuda', type=int, default=7,
                        help='cuda device.')
    parser.add_argument('--l2_decay', type=float, default=1e-3,
                        help='l2 loss reg coef.')
    parser.add_argument('--alpha', type=float, default=0,
                        help='dro alpha.')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='for robust radius')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='dropout ')
    parser.add_argument('--descri', type=str, default='',
                        help='description of the work.')
    parser.add_argument("--random_sample", type=int, default=100, help="the random sample num")
    parser.add_argument('--seed', type=int, default=6,
                        help='Random seed.')
    parser.add_argument("--log_file", type=str, default="caser_result.log")
    return parser.parse_args()



class GRU(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, gru_layers=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = item_num
        self.state_size = state_size
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=gru_layers,
            batch_first=True
        )
        self.s_fc = nn.Linear(self.hidden_size, self.item_num)

    def forward(self, states, len_states):
        # Supervised Head
        emb = self.item_embeddings(states)
        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(emb, len_states, batch_first=True, enforce_sorted=False)
        emb_packed, hidden = self.gru(emb_packed)
        hidden = hidden.view(-1, hidden.shape[2])
        supervised_output = self.s_fc(hidden)
        return supervised_output

    def forward_eval(self, states, len_states):
        # Supervised Head
        emb = self.item_embeddings(states)
        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(emb, len_states, batch_first=True, enforce_sorted=False)
        emb_packed, hidden = self.gru(emb_packed)
        hidden = hidden.view(-1, hidden.shape[2])
        supervised_output = self.s_fc(hidden)

        return supervised_output


class Caser(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, num_filters, filter_sizes,
                 dropout_rate):
        super(Caser, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.state_size = state_size
        self.filter_sizes = eval(filter_sizes)
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.hidden_size,
        )

        # init embedding
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)

        # Horizontal Convolutional Layers
        self.horizontal_cnn = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (i, self.hidden_size)) for i in self.filter_sizes])
        # Initialize weights and biases
        for cnn in self.horizontal_cnn:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        # Vertical Convolutional Layer
        self.vertical_cnn = nn.Conv2d(1, 1, (self.state_size, 1))
        nn.init.xavier_normal_(self.vertical_cnn.weight)
        nn.init.constant_(self.vertical_cnn.bias, 0.1)

        # Fully Connected Layer
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        final_dim = self.hidden_size + self.num_filters_total
        self.s_fc = nn.Linear(final_dim, item_num)

        # dropout
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, states, len_states):
        input_emb = self.item_embeddings(states)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1)
        input_emb *= mask
        input_emb = input_emb.unsqueeze(1)
        pooled_outputs = []
        for cnn in self.horizontal_cnn:
            h_out = nn.functional.relu(cnn(input_emb))
            h_out = h_out.squeeze()
            p_out = nn.functional.max_pool1d(h_out, h_out.shape[2])
            pooled_outputs.append(p_out)

        h_pool = torch.cat(pooled_outputs, 1)
        h_pool_flat = h_pool.view(-1, self.num_filters_total)

        v_out = nn.functional.relu(self.vertical_cnn(input_emb))
        v_flat = v_out.view(-1, self.hidden_size)

        out = torch.cat([h_pool_flat, v_flat], 1)
        out = self.dropout(out)
        supervised_output = self.s_fc(out)

        return supervised_output

    def forward_eval(self, states, len_states):
        input_emb = self.item_embeddings(states)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1)
        input_emb *= mask
        input_emb = input_emb.unsqueeze(1)
        pooled_outputs = []
        for cnn in self.horizontal_cnn:
            h_out = nn.functional.relu(cnn(input_emb))
            h_out = h_out.squeeze()
            p_out = nn.functional.max_pool1d(h_out, h_out.shape[2])
            pooled_outputs.append(p_out)

        h_pool = torch.cat(pooled_outputs, 1)
        h_pool_flat = h_pool.view(-1, self.num_filters_total)

        v_out = nn.functional.relu(self.vertical_cnn(input_emb))
        v_flat = v_out.view(-1, self.hidden_size)

        out = torch.cat([h_pool_flat, v_flat], 1)
        out = self.dropout(out)
        supervised_output = self.s_fc(out)
        
        return supervised_output

class Caser_with_label(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, num_filters, filter_sizes,
                 dropout_rate):
        super(Caser_with_label, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.state_size = state_size
        self.filter_sizes = eval(filter_sizes)
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        # 这里留下一维用于rate
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.hidden_size - 1,
        )

        # init embedding
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)

        # Horizontal Convolutional Layers
        self.horizontal_cnn = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (i, self.hidden_size)) for i in self.filter_sizes])
        # Initialize weights and biases
        for cnn in self.horizontal_cnn:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        # Vertical Convolutional Layer
        self.vertical_cnn = nn.Conv2d(1, 1, (self.state_size, 1))
        nn.init.xavier_normal_(self.vertical_cnn.weight)
        nn.init.constant_(self.vertical_cnn.bias, 0.1)

        # Fully Connected Layer
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        final_dim = self.hidden_size + self.num_filters_total
        self.s_fc = nn.Linear(final_dim, item_num)

        # dropout
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, states, state_rate, len_states):
        input_emb = self.item_embeddings(states)
        state_rate_reshape = state_rate.view(-1,10,1)
        input_emb = torch.cat((input_emb, state_rate_reshape), dim=2)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1)
        input_emb *= mask
        input_emb = input_emb.unsqueeze(1)
        pooled_outputs = []
        for cnn in self.horizontal_cnn:
            h_out = nn.functional.relu(cnn(input_emb))
            h_out = h_out.squeeze()
            p_out = nn.functional.max_pool1d(h_out, h_out.shape[2])
            pooled_outputs.append(p_out)

        h_pool = torch.cat(pooled_outputs, 1)
        h_pool_flat = h_pool.view(-1, self.num_filters_total)

        v_out = nn.functional.relu(self.vertical_cnn(input_emb))
        v_flat = v_out.view(-1, self.hidden_size)

        out = torch.cat([h_pool_flat, v_flat], 1)
        out = self.dropout(out)
        supervised_output = self.s_fc(out)

        return supervised_output

    def forward_eval(self, states, len_states):
        input_emb = self.item_embeddings(states)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1)
        input_emb *= mask
        input_emb = input_emb.unsqueeze(1)
        pooled_outputs = []
        for cnn in self.horizontal_cnn:
            h_out = nn.functional.relu(cnn(input_emb))
            h_out = h_out.squeeze()
            p_out = nn.functional.max_pool1d(h_out, h_out.shape[2])
            pooled_outputs.append(p_out)

        h_pool = torch.cat(pooled_outputs, 1)
        h_pool_flat = h_pool.view(-1, self.num_filters_total)

        v_out = nn.functional.relu(self.vertical_cnn(input_emb))
        v_flat = v_out.view(-1, self.hidden_size)

        out = torch.cat([h_pool_flat, v_flat], 1)
        out = self.dropout(out)
        supervised_output = self.s_fc(out)
        
        return supervised_output

def evaluate_auc_with_history_label(model, test_path, device):
    model = model.eval()
    # test_path = "test.csv"
    test_data = pd.read_csv(test_path)
    seq, seq_rate, target, target_label = change_batch_label_book(test_data, max_len = 10)
    with torch.no_grad():
        seq = torch.LongTensor(seq)
        target = torch.LongTensor(target)
        seq_rate = torch.LongTensor(seq_rate)
        seq = seq.to(device)
        target = target.to(device)
        seq_rate = seq_rate.to(device)
        model_output = nn.Sigmoid()(model.forward(seq, seq_rate, max_len))
        target = target.view(-1, 1)
        scores = torch.gather(model_output, 1, target)
    scores = scores.view(-1)
    auc = roc_auc_score(target_label, scores.cpu())
    # scores_copy = copy.deepcopy(scores)
    # # acc_list = []
    # # for thresh in thresh_list:
    # #     orig_scores = copy.deepcopy(scores)
    # #     orig_scores[scores_copy > thresh] = 1
    # #     orig_scores[scores_copy < thresh] = 0
    # #     acc_list.append(torch.sum((orig_scores.cpu() == torch.tensor(target_label))).item()/len(target_label))
    return auc


def change_batch_label_book(batch, max_len):
    """
    这个是用于考虑当sequential的时候可以将label作为state的一部分输入进去用的
    """
    history_id = batch["history_item_id"]
    history_rating = batch["history_rating"]
    target = batch["item_id"]
    rating = batch["rating"]
    final_history_list= np.array([])
    final_target_list = []
    final_target_label = []
    final_history_rate = []
    for key in history_id.keys():
        history_id_temp = np.array(eval(history_id[key]), dtype="int")
        history_rating_temp = np.array(eval(history_rating[key]), dtype="int")
        target_temp = target[key]
        rating_temp = rating[key]
        # if sum(history_rating_temp) == 0:
        #     continue
        # pos_his = np.array(history_id_temp[history_rating_temp==1])
        # neg_his = np.array(history_id_temp[history_rating_temp==0])
        history_id_temp = np.pad(history_id_temp, pad_width=(0, max_len - len(history_id_temp)), mode="constant", constant_values=history_id_temp[-1]).reshape(1,-1)
        history_rating_temp = np.pad(history_rating_temp, pad_width=(0, max_len - len(history_rating_temp)), mode="constant", constant_values=history_rating_temp[-1]).reshape(1,-1)
        if len(final_target_list) == 0:
            final_history_list = history_id_temp.reshape(1,-1)
            final_history_rate = history_rating_temp.reshape(1,-1)
        else:
            final_history_list = np.concatenate((final_history_list, history_id_temp.reshape(1,-1)))
            final_history_rate = np.concatenate((final_history_rate, history_rating_temp.reshape(1,-1)))
        final_target_list.append(target_temp)
        final_target_label.append(rating_temp)
    return final_history_list, final_history_rate, final_target_list, final_target_label



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    result_list_valid = []
    result_list_test = []
    sample_list = [1,2,4,8,16,32,64,128,256,512, 19414]
    for sample_num in sample_list:
        args = parse_args()
        setup_seed(args.seed)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

        # logging.basicConfig(filename="./log/{}/{}_{}_lr{}_decay{}_dro{}_gamma{}_beta{}".format(args.data + '_final2', Time.strftime("%m-%d %H:%M:%S", Time.localtime()), args.model_name, args.lr, args.l2_decay, args.dro_reg, args.gamma, args.beta))
        # Network parameters

        data_directory = './data/' + args.data
        # data_directory = './data/' + args.data
        # data_directory = '../' + args.data + '/data'
        # data_statis = pd.read_pickle(
        #     os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing seq_size and item_num

        train_data = pd.read_csv("train.csv")
        train_data = train_data.sample(n=sample_num,random_state=args.seed)
        
        max_len = 10
        seq_size = max_len  # the length of history to define the seq
        item_num = 271380 + 1  # total number of items
        thresh_list = [0.00005, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        # topk=[10, 20, 50]


        model = Caser_with_label(args.hidden_factor,item_num, seq_size, args.num_filters, args.filter_sizes, args.dropout_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
        mse_loss = nn.MSELoss()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # optimizer.to(device)

        # train_data = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))
        # ps = calcu_propensity_score(train_data)
        # ps = torch.tensor(ps)
        # ps = ps.to(device)

        total_step=0
        hr_max = 0
        best_epoch = 0

        num_rows=train_data.shape[0]
        num_batches=int(num_rows/args.batch_size)
        if args.batch_size > num_rows:
            args.batch_size = num_rows
            num_batches = 1
        # best_valid_acc = 0
        # best_test_acc = 0
        best_valid_auc = 0
        best_test_auc = 0
        for i in range(args.epoch):
            for j in range(num_batches):
                model = model.train()
                batch = train_data.sample(n=args.batch_size).to_dict()

                seq, seq_rate, target, target_label = change_batch_label_book(batch, max_len=max_len)
                optimizer.zero_grad()
                seq = torch.LongTensor(seq)
                seq_rate = torch.FloatTensor(seq_rate)
                # len_seq = torch.LongTensor(len_seq)
                target = torch.LongTensor(target)
                # target_neg = torch.LongTensor(target_neg)
                seq = seq.to(device)
                target = target.to(device)
                seq_rate = seq_rate.to(device)


                model_output = model.forward(seq, seq_rate, max_len).view(-1, item_num)


                target = target.view(-1, 1)

                scores = torch.gather(model_output, 1, target)

                labels = torch.tensor(target_label).to(device).view(-1,1)

                scores = nn.Sigmoid()(scores)

                loss = mse_loss(scores, labels.float())

                loss.backward()
                optimizer.step()

                if True:

                    total_step+=1
 

                    if total_step % 5 == 0:
                            valid_auc = evaluate_auc_with_history_label(model, 'valid.csv', device)
                            test_auc = evaluate_auc_with_history_label(model, 'test.csv', device)

                            if best_valid_auc < valid_auc:
                                best_valid_auc = valid_auc
                                best_test_auc = test_auc
                            
                            print("Best test auc:" + str(best_test_auc))

        result_list_valid.append(best_valid_auc)
        result_list_test.append(best_test_auc)
        print("valid")
        print(result_list_valid)
        print("test")
        print(result_list_test)
    with open(args.log_file, "a+") as f:
        f.write("random seed:" + str(args.seed) + "\n")
        f.write("sample list:\n")
        f.write(str(sample_list) + "\n")
        f.write("valid:\n")
        f.write(str(result_list_valid) + "\n")
        f.write("test:\n")
        f.write(str(result_list_test) + "\n")

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     

