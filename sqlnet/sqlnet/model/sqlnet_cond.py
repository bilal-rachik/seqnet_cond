import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sqlnet.model.modules.word_embedding import WordEmbedding
from sqlnet.model.modules.sqlnet_condition_predict_cond import SQLNetCondPredictor_cond


class SQLNet_cond(nn.Module):
    def __init__(self, word_emb, N_word, N_h=100, N_depth=2,gpu=False, use_ca=True, trainable_emb=False):
        super(SQLNet_cond, self).__init__()
        self.use_ca = use_ca
        self.trainable_emb = trainable_emb
        self.gpu = gpu
        self.N_h = N_h
        self.N_depth = N_depth
        self.max_col_num = 45
        self.max_tok_num = 200
        self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND',
                'EQL', 'GT', 'LT', '<BEG>']
        self.COND_OPS = ['EQL', 'GT', 'LT']
        
        #Word embedding
        if trainable_emb:
           
            self.cond_embed_layer = WordEmbedding(word_emb, N_word, gpu,
                    self.SQL_TOK, our_model=True, trainable=trainable_emb)
        else:
            self.embed_layer = WordEmbedding(word_emb, N_word, gpu,
                    self.SQL_TOK, our_model=True, trainable=trainable_emb)
        
       
        
        #Predict number of cond
        self.cond_pred = SQLNetCondPredictor_cond(N_word,N_h,N_depth,
                self.max_col_num,self.max_tok_num,use_ca,gpu)
        
        
        
        self.CE = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()
        self.log_softmax = nn.LogSoftmax()
        self.bce_logit = nn.BCEWithLogitsLoss()
        
        
        
        
    def generate_gt_where_seq(self,q, query):
        ret_seq = []

        for cur_q,cur_query in zip(q,query):
            cur_values = []
            
            st = cur_query.index(u'WHERE')+1 if \
                    u'WHERE' in cur_query else len(cur_query)
            all_toks = ['<BEG>'] + cur_q + ['<END>']
            while st < len(cur_query):
                ed = len(cur_query) if 'AND' not in cur_query[st:]\
                        else cur_query[st:].index('AND') + st
                if 'EQL' in cur_query[st:ed]:
                    op = cur_query[st:ed].index('EQL') + st
                elif 'GT' in cur_query[st:ed]:
                    op = cur_query[st:ed].index('GT') + st
                elif 'LT' in cur_query[st:ed]:
                    op = cur_query[st:ed].index('LT') + st
                else:
                    raise RuntimeError("No operator in it!")
                this_str = ['<BEG>'] + cur_query[op+1:ed] + ['<END>']
                cur_seq = [all_toks.index(s) if s in all_toks \
                        else 0 for s in this_str]
                cur_values.append(cur_seq)
                st = ed+1
            ret_seq.append(cur_values)
        return ret_seq


    def forward(self, q, col, col_num,gt_where = None, gt_cond=None, reinforce=False):
        
        cond_score = None

        #Predict aggregator
        if self.trainable_emb:
            x_emb_var, x_len = self.cond_embed_layer.gen_x_batch(q, col)
            
            col_inp_var, col_name_len, col_len = \
                    self.cond_embed_layer.gen_col_batch(col)
                    
            cond_score = self.cond_pred(x_emb_var, x_len, col_inp_var,
                    col_name_len, col_len, col_num,
                    gt_where, gt_cond)
        else:
            x_emb_var,x_len = self.embed_layer.gen_x_batch(q,col)
            
            col_inp_var, col_name_len, col_len = \
                    self.embed_layer.gen_col_batch(col)

            cond_score = self.cond_pred(x_emb_var, x_len, col_inp_var,
                    col_name_len, col_len, col_num,
                    gt_where, gt_cond)

        return cond_score

    def loss(self, score, truth_num,gt_where):
        loss = 0
        B = len(truth_num)
        cond_num_score, cond_col_score,\
                cond_op_score, cond_str_score = score
        #Evaluate the number of conditions
        cond_num_truth = map(lambda x:x[2], truth_num)
        aa=list(cond_num_truth)
        data = torch.from_numpy(np.array(aa))
        if self.gpu:
            cond_num_truth_var = Variable(data.cuda())
        else:
            cond_num_truth_var = Variable(data)
            #CE=nn.CrossEntropyLoss()
        loss += self.CE(cond_num_score, cond_num_truth_var.long())

        #Evaluate the columns of conditions
        T = len(cond_col_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            if len(truth_num[b][3]) > 0:
                truth_prob[b][list(truth_num[b][3])] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            cond_col_truth_var = Variable(data.cuda())
        else:
            cond_col_truth_var = Variable(data)

        sigm = nn.Sigmoid()
        cond_col_prob = sigm(cond_col_score)
        bce_loss = -torch.mean( 3*(cond_col_truth_var * \
                torch.log(cond_col_prob+1e-10)) + \
                (1-cond_col_truth_var) * torch.log(1-cond_col_prob+1e-10) )
        loss += bce_loss

        #Evaluate the operator of conditions
        for b in range(len(truth_num)):
            if len(truth_num[b][4]) == 0:
                continue
            data = torch.from_numpy(np.array(truth_num[b][4]))
            if self.gpu:
                cond_op_truth_var = Variable(data.cuda())
            else:
                cond_op_truth_var = Variable(data)
            cond_op_pred = cond_op_score[b, :len(truth_num[b][4])]
            loss += (self.CE(cond_op_pred, cond_op_truth_var.long()) \
                    / len(truth_num))

        #Evaluate the strings of conditions
        for b in range(len(gt_where)):
            for idx in range(len(gt_where[b])):
                cond_str_truth = gt_where[b][idx]
                if len(cond_str_truth) == 1:
                    continue
                data = torch.from_numpy(np.array(cond_str_truth[1:]))
                if self.gpu:
                    cond_str_truth_var = Variable(data.cuda())
                else:
                    cond_str_truth_var = Variable(data)
                str_end = len(cond_str_truth)-1
                cond_str_pred = cond_str_score[b, idx, :str_end]
                loss += (self.CE(cond_str_pred, cond_str_truth_var.long()) \
                        / (len(gt_where) * len(gt_where[b])))

        return loss

    def check_acc(self, vis_info, pred_queries, gt_queries):
        def pretty_print(vis_data):
            print ('question:', vis_data[0])
            print ('headers: (%s)'%(' || '.join(vis_data[1])))
            print ('query:', vis_data[2])

        def gen_cond_str(conds, header):
            if len(conds) == 0:
                return 'None'
            cond_str = []
            for cond in conds:
                cond_str.append(header[cond[0]] + ' ' +
                    self.COND_OPS[cond[1]] + ' ' + unicode(cond[2]).lower())
            return 'WHERE ' + ' AND '.join(cond_str)
        B = len(gt_queries)

        tot_err= cond_err = 0.0
        cond_num_err = cond_col_err = cond_op_err = cond_val_err = 0.0
        
        for b, (pred_qry, gt_qry) in enumerate(zip(pred_queries, gt_queries)):
            good = True 
            cond_pred = pred_qry['conds']
            cond_gt = gt_qry['conds']
            flag = True
            if len(cond_pred) != len(cond_gt):
                flag = False
                cond_num_err += 1

            if flag and set(x[0] for x in cond_pred) != \
                    set(x[0] for x in cond_gt):
                flag = False
                cond_col_err += 1

            for idx in range(len(cond_pred)):
                if not flag:
                    break
                gt_idx = tuple(
                        x[0] for x in cond_gt).index(cond_pred[idx][0])
                if flag and cond_gt[gt_idx][1] != cond_pred[idx][1]:
                    flag = False
                    cond_op_err += 1

            for idx in range(len(cond_pred)):
                if not flag:
                    break
                gt_idx = tuple(
                        x[0] for x in cond_gt).index(cond_pred[idx][0])
                if flag and str(cond_gt[gt_idx][2]).lower() != \
                        str(cond_pred[idx][2]).lower():
                    flag = False
                    cond_val_err += 1

            if not flag:
                cond_err += 1
                good = False

            if not good:
                tot_err += 1

        return np.array((cond_err)), tot_err
    
    


    def gen_query(self, score, q, raw_q, verbose=False):
        def merge_tokens(tok_list, raw_tok_str):
            tok_str=[word.lower() for word in raw_tok_str]
            #tok_str = raw_tok_str.lower()
            alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789$('
            special = {'-LRB-':'(',
                    '-RRB-':')',
                    '-LSB-':'[',
                    '-RSB-':']',
                    '``':'"',
                    '\'\'':'"',
                    '--':u'\u2013'}
            ret = ''
            double_quote_appear = 0
            for raw_tok in tok_list:
                if not raw_tok:
                    continue
                tok = special.get(raw_tok, raw_tok)
                if tok == '"':
                    double_quote_appear = 1 - double_quote_appear

                if len(ret) == 0:
                    pass
                elif len(ret) > 0 and ret + ' ' + tok in tok_str:
                    ret = ret + ' '
                elif len(ret) > 0 and ret + tok in tok_str:
                    pass
                elif tok == '"':
                    if double_quote_appear:
                        ret = ret + ' '
                elif tok[0] not in alphabet:
                    pass
                elif (ret[-1] not in ['(', '/', u'\u2013', '#', '$', '&']) \
                        and (ret[-1] != '"' or not double_quote_appear):
                    ret = ret + ' '
                ret = ret + tok
            return ret.strip()

        cond_score = score

        ret_queries = []
        
        
        B = len(cond_score[0])
        for b in range(B):
            cur_query = {}
            
            
        
            cur_query['conds'] = []
            cond_num_score,cond_col_score,cond_op_score,cond_str_score =\
                    [x.data.cpu().numpy() for x in cond_score]
            cond_num = np.argmax(cond_num_score[b])
            all_toks = ['<BEG>'] + q[b] + ['<END>']
            max_idxes = np.argsort(-cond_col_score[b])[:cond_num]
            for idx in range(cond_num):
                cur_cond = []
                cur_cond.append(max_idxes[idx])
                cur_cond.append(np.argmax(cond_op_score[b][idx]))
                cur_cond_str_toks = []
                for str_score in cond_str_score[b][idx]:
                    str_tok = np.argmax(str_score[:len(all_toks)])
                    str_val = all_toks[str_tok]
                    if str_val == '<END>':
                        break
                    cur_cond_str_toks.append(str_val)
                cur_cond.append(merge_tokens(cur_cond_str_toks, raw_q[b]))
                cur_query['conds'].append(cur_cond)
            ret_queries.append(cur_query)

        return ret_queries
  


