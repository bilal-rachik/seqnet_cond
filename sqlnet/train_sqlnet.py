### TODO: Need to be test yet!
import json
import torch
from sqlnet.utils_sqlnet import *
from sqlnet.model.sqlnet_cond import SQLNet_cond
import numpy as np
import datetime


N_word=300
B_word=42   
BATCH_SIZE=64
USE_SMALL=False
GPU=False     
learning_rate =1e-3

sql_data, table_data, val_sql_data, val_table_data,\
        test_sql_data, test_table_data, TRAIN_DB, DEV_DB, TEST_DB = \
        load_dataset(0, use_small=USE_SMALL)
            
word_emb = load_word_emb('glove/glove.42B.300d.txt', \
            load_used=True, use_small=USE_SMALL)

model_sqlnet_cond= SQLNet_cond(word_emb, N_word=N_word, use_ca=True,
                gpu=GPU,trainable_emb =True)

cond_e="sqlnet_emb_cond"
cond_m="sqlnet_cond"


model_sqlnet_cond.cond_pred.load_state_dict(torch.load(cond_m))
model_sqlnet_cond.cond_embed_layer.load_state_dict(torch.load(cond_e))

optimizer = torch.optim.Adam(model_sqlnet_cond.parameters(),
            lr=learning_rate, weight_decay = 0)


init_acc = epoch_acc_sqlnet(model_sqlnet_cond, BATCH_SIZE,
                val_sql_data, val_table_data)



best_cond_acc = init_acc[0]
best_cond_idx = 0


print ('Init dev acc_qm: %s\n  breakdown on (where): %s'%\
                init_acc)
torch.save(model_sqlnet_cond.cond_pred.state_dict(), cond_e)
           
for i in range(69,100):
    print ('Epoch %d @ %s'%(i+1, datetime.datetime.now()))
    print (' Loss = %s'%epoch_train_sqlnet(
            model_sqlnet_cond,optimizer, BATCH_SIZE, 
            sql_data, table_data))
    
    print (' Train acc_qm: %s\n   breakdown result: %s'%epoch_acc_sqlnet(
            model_sqlnet_cond, BATCH_SIZE, sql_data, table_data,))
    val_acc =epoch_acc_sqlnet(model_sqlnet_cond,
            BATCH_SIZE, val_sql_data, val_table_data)
    print (' Dev acc_qm: %s\n   breakdown result: %s'%val_acc)
   
    if val_acc[0]> best_cond_acc:
        best_cond_acc = val_acc[0]
        best_cond_idx = i+1
        torch.save(model_sqlnet_cond.cond_pred.state_dict(),
                        'epoch%d.cond_model%s'%(i+1,cond_m))
        torch.save(model_sqlnet_cond.cond_pred.state_dict(), cond_m)
                    
        torch.save(model_sqlnet_cond.cond_embed_layer.state_dict(),
                        'epoch%d.cond_embed%s'%(i+1,cond_e))
        torch.save(model_sqlnet_cond.cond_embed_layer.state_dict(), cond_e)
            
    print (' Best val acc = %s, on epoch %s individually'%(
            (best_cond_acc),
            (best_cond_idx)))








