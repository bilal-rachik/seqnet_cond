import json
from sqlnet.lib.dbengine import DBEngine
import re
import numpy as np

def load_data(sql_paths, table_paths, use_small=False):
    if not isinstance(sql_paths, list):
        sql_paths = (sql_paths, )
    if not isinstance(table_paths, list):
        table_paths = (table_paths, )
    sql_data = []
    table_data = {}

    
    for SQL_PATH in sql_paths:
        print ("Loading data from %s"%SQL_PATH)
        with open(SQL_PATH) as inf:
            for idx, line in enumerate(inf):
                if use_small and idx >= 1000:
                    break
                sql = json.loads(line.strip())
                sql_data.append(sql)

    for TABLE_PATH in table_paths:
        print ("Loading data from %s"%TABLE_PATH)
        with open(TABLE_PATH) as inf:
            for line in inf:
                tab = json.loads(line.strip())
                table_data[tab[u'id']] = tab

    for sql in sql_data:
        assert sql[u'table_id'] in table_data

    return sql_data, table_data


def load_dataset(dataset_id, use_small=False):
    if dataset_id == 0:
        print ("Loading from original dataset")
        sql_data,table_data = load_data('data/train_tok.jsonl',
                'data/train_tok.tables.jsonl',use_small=use_small)
        val_sql_data, val_table_data = load_data('data/dev_tok.jsonl',
                'data/dev_tok.tables.jsonl', use_small=use_small)
        test_sql_data, test_table_data = load_data('data/test_tok.jsonl',
                'data/test_tok.tables.jsonl', use_small=use_small)
        TRAIN_DB = 'data/train.db'
        DEV_DB = 'data/dev.db'
        TEST_DB = 'data/test.db'
    else:
        print ("Loading from re-split dataset")
        sql_data, table_data = load_data('data_resplit/train.jsonl',
                'data_resplit/tables.jsonl', use_small=use_small)
        val_sql_data, val_table_data = load_data('data_resplit/dev.jsonl',
                'data_resplit/tables.jsonl', use_small=use_small)
        test_sql_data, test_table_data = load_data('data_resplit/test.jsonl',
                'data_resplit/tables.jsonl', use_small=use_small)
        TRAIN_DB = 'data_resplit/table.db'
        DEV_DB = 'data_resplit/table.db'
        TEST_DB = 'data_resplit/table.db'

    return sql_data, table_data, val_sql_data, val_table_data,\
            test_sql_data, test_table_data, TRAIN_DB, DEV_DB, TEST_DB

def to_batch_seq(sql_data,table_data, idxes, st, ed, ret_vis_data=False):
    q_seq = []
    col_seq = []
    col_num = []
    ans_seq = []
    query_seq = []
    gt_cond_seq = []
    vis_seq = []
    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        q_seq.append(sql['question_tok'])
        col_seq.append(table_data[sql['table_id']]['header_tok'])
        col_num.append(len(table_data[sql['table_id']]['header_tok']))
        ans_seq.append((sql['sql']['agg'],
            sql['sql']['sel'], 
            len(sql['sql']['conds']),
            tuple(x[0] for x in sql['sql']['conds']),
            tuple(x[1] for x in sql['sql']['conds'])))
        
        query_seq.append(sql['query_tok'])
        
        gt_cond_seq.append(sql['sql']['conds'])
        vis_seq.append((sql['question_tok'],
            table_data[sql['table_id']]['header'], sql['query_tok']))
    if ret_vis_data:
        return q_seq, col_seq,col_num,ans_seq,query_seq, gt_cond_seq, vis_seq
    else:
        return q_seq,col_seq,col_num,ans_seq,query_seq, gt_cond_seq

def to_batch_query(sql_data, idxes, st, ed):
    query_gt = []
    table_ids = []
    for i in range(st, ed):
        query_gt.append(sql_data[idxes[i]]['sql'])
        table_ids.append(sql_data[idxes[i]]['table_id'])
    return query_gt, table_ids

def epoch_train_sqlnet(model_sqlnet_cond,optimizer,batch_size, sql_data, table_data):
    model_sqlnet_cond.train()
    perm=np.random.permutation(len(sql_data))
    cum_loss = 0.0
    st = 0
    while st < len(sql_data):
        ed= st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq = \
                to_batch_seq(sql_data,table_data, perm, st, ed)
                
        gt_where_seq =model_sqlnet_cond.generate_gt_where_seq(q_seq,query_seq)
        
        score= model_sqlnet_cond.forward(q_seq,col_seq, col_num,
                gt_where=gt_where_seq,gt_cond=gt_cond_seq)
        
        loss = model_sqlnet_cond.loss(score, ans_seq,gt_where_seq)

    
        cum_loss += loss.data.cpu().numpy()*(ed - st)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        st = ed

    return cum_loss / len(sql_data)

def epoch_acc_sqlnet(model_sqlnet_cond, batch_size, sql_data, table_data):
    model_sqlnet_cond.eval()
    perm = list(range(len(sql_data)))
    st = 0
    one_acc_num = 0.0
    tot_acc_num = 0.0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data = to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        score = model_sqlnet_cond.forward(q_seq,col_seq,col_num)
        
        pred_queries = model_sqlnet_cond.gen_query(score,q_seq,raw_q_seq)
        
        one_err, tot_err = model_sqlnet_cond.check_acc(raw_data,pred_queries,query_gt)

        one_acc_num += (ed-st-one_err)
        tot_acc_num += (ed-st-tot_err)

        st = ed
    return tot_acc_num / len(sql_data), one_acc_num / len(sql_data)

def load_word_emb(file_name, load_used=False, use_small=False):
    if not load_used:
        print ('Loading word embedding from %s'%file_name)
        word2emb = {}
        i=0
        fglove=open(file_name,"rb")
        for line in fglove:
            cols = line.strip().split()
            word = cols[0].decode('utf-8')
            embedding = np.array(cols[1:], dtype="float32")
            word2emb[word] = embedding
        fglove.close()
        return word2emb
    else:
        print ('Load used word embedding')
        with open('glove/word2idx.json') as inf:
            w2i = json.load(inf)
        with open('glove/usedwordemb.npy',"rb") as inf:
            word_emb_val = np.load(inf)
        return w2i, word_emb_val

