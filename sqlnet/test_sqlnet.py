import json
import torch
from sqlnet.utils_sqlnet import *
from sqlnet.model.sqlnet_cond import SQLNet_cond
import numpy as np
import datetime


N_word=300
B_word=42
USE_SMALL=False
GPU=False
BATCH_SIZE=64
   

sql_data, table_data, val_sql_data, val_table_data, \
            test_sql_data, test_table_data, \
            TRAIN_DB, DEV_DB, TEST_DB = load_dataset(0, use_small=USE_SMALL)

word_emb = load_word_emb('data/glove.42B.300d.txt', \
            load_used=True, use_small=USE_SMALL)

model=SQLNet_cond(word_emb, N_word=N_word, use_ca=True, gpu=GPU,
                trainable_emb = True)



cond_e="sqlnet_emb_cond"
cond_m="sqlnet_cond"

print ("Loading from %s"%cond_m)
model.cond_pred.load_state_dict(torch.load(cond_m))
print ("Loading from %s"%cond_e)
model.cond_embed_layer.load_state_dict(torch.load(cond_e))

print ("Dev acc_qm: %s;\n  breakdown on (where): %s"%epoch_acc_sqlnet(
            model, BATCH_SIZE, val_sql_data, val_table_data))

print ("Test acc_qm: %s;\n  breakdown on (agg, sel, where): %s"%epoch_acc_sqlnet(
            model, BATCH_SIZE, test_sql_data, test_table_data))
