#!/usr/bin/env python
# coding: utf-8


import tqdm
import mlflow
import json
import statistics
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from KGE.models.translating_based.TransE import TransE


def batch(iterable, n = 1):
    # generate batches of batch_size:n 
    current_batch = []
    for item in iterable:
        current_batch.append(item)
        if len(current_batch) == n:
            yield current_batch
            current_batch = []
    if current_batch:
        yield current_batch
        
def recommend(user_list):
    '''
    A function to recommend 25 musics for each user in the input user list

        Parameter
        ---------
            user_list: list of user id
        
        Return
        ------
            dict: top 25 recommend songs for list of users
    '''
    # - input: list of user id
    # - output: list of recommend item (25 recommend songs for each user)
    # - logic:
    #     1. user id → user embedding
    #     2. a = user embedding + has_insterest embedding
    #     3. compare distance with all item embeddings, output the nearest 25 items

    test_users_rec_music = {}
    for users in tqdm.tqdm(batch(user_list,100), total=len(user_list)//100+1):
        # users embedding (batch_users * embedding_size)
        users_index = [metadata['ent2ind'].get(user) for user in users]
        users_emb = tf.nn.embedding_lookup(model.model_weights['ent_emb'], users_index)

        # has_interest embedding (1 * embedding_size )
        has_interest_index = metadata['rel2ind']['has_interest']
        has_interest_emb = model.model_weights['rel_emb'][has_interest_index]
        
        # compute recommend songs (batch_users * embedding_size)
        compute_songs_emb = users_emb + has_interest_emb

        with open('./data/KKBOX/entity_groupby_type.json') as f:
            entity_groupby_type = json.load(f)

        # songs embedding (total_songs * embedding_size)
        song_id = [metadata['ent2ind'].get(ent) for ent in entity_groupby_type['song']]
        songs_emb = tf.nn.embedding_lookup(model.model_weights['ent_emb'], song_id)

        # 用matrix計算，算完全部compute_songs_emb (list) 與 全部songs_emb(list)的距離 (batch_users * total_songs)
        distances = [] 
        # for each user
        for i in range(compute_songs_emb.shape[0]):
            # calculate his rec_music embedding distance to all songs embeddings
            distances.append(tf.norm(tf.subtract(songs_emb, compute_songs_emb[i]), ord=2, axis=1))

        # 每個人的前25首embedding相似的song index (batch_users * 25)
        top_25_songs_index = tf.argsort(distances)[:,:25].numpy().tolist() 

        # song index to song id (batch_users * 25)
        song_ent = tf.convert_to_tensor(np.array(entity_groupby_type['song']))
        top_25_songs = tf.nn.embedding_lookup(song_ent, top_25_songs_index)

        # zip users and their rec_25_songs into a dict
        users_top25_songs =  dict(zip(users,top_25_songs))
        test_users_rec_music.update(users_top25_songs)
    
    return test_users_rec_music


# NDCG

def DCG(rec_list, ans_list):
    dcg = 0
    for i in range(len(rec_list)):
        r_i = 0
        if rec_list[i] in ans_list:
            r_i = 1
        dcg += (2**r_i - 1) / np.log2((i + 1) + 1)
    return dcg

def IDCG(rec_list, ans_list):
    A_temp_1 = []
    A_temp_0 = []
    for rec_music in rec_list:
        if rec_music in ans_list:
            A_temp_1.append(rec_music)
        else:
            A_temp_0.append(rec_music)
    A_temp_1.extend(A_temp_0)
    idcg = DCG(A_temp_1, ans_list)
    return idcg

def NDCG(rec_list, ans_list):
    dcg = DCG(rec_list, ans_list)
    idcg = IDCG(rec_list, ans_list)
    if dcg == 0 or idcg ==0:
        ndcg = 0
    else:
        ndcg = dcg / idcg
    return ndcg
    
def intersection(list1, list2):
    # check if two lists have intersect
    return list(set(list1) & set(list2))
    
def evaluate(test_users_rec_music):
    '''
    Evaluate the recommend result
        
        Parameters
        ----------
            test_users_rec_music(dict): top 25 recommended songs for each user
            log_path: the path to write in tensorboard log

        Returns
        -------
            metric_result(dict): metric include hit, recall, precision and NDCG
    '''
    TP_list = [] # each user's True Positive number
    ans_lengths = [] # each user's has_interest music number
    ndcg_list = []
    for user in test_users_rec_music.keys():
        ans_music_list = user_and_hasInterestItem[user]
        ans_lengths.append(len(ans_music_list))
        rec_music_list = [x.decode() for x in test_users_rec_music[user].numpy().tolist()]
        TP_list.append(len(intersection(rec_music_list, ans_music_list)))
        ndcg_list.append(NDCG(rec_music_list, ans_music_list))
        
    hit_list = [1 if TP >= 1 else 0 for TP in TP_list]
    precision_list = [TP/25 for TP in TP_list]
    recall_list = [TP_list[i]/ans_lengths[i] for i in range(len(TP_list))]

    metric_result = {
        'hit': statistics.mean(hit_list),
        'recall': statistics.mean(recall_list),
        'precision': statistics.mean(precision_list),
        'ndcg': statistics.mean(ndcg_list)
    }

    return metric_result
    

def generateTestData(df):
    '''
        Parameter
        ---------
            df: dataframe
        
        Return
        ------
            users: test users list
            user_and_hasInterestItem: dict{key=user: value=interest item list}
    '''
    users = df['h'].unique().tolist()
    user_and_hasInterestItem = df.groupby('h')['t'].apply(list).to_dict()
    return users, user_and_hasInterestItem



mlflow.set_experiment('KKBOX-MusicRecommend')

emb_size = 30
neg_ratio = 32
epoch = 100

run_name = 'TEST-emb' + str(emb_size) + 'neg' + str(neg_ratio) + 'epoch' + str(epoch)
log_path = './tensorboard_logs/TEST-emb' + str(emb_size) + 'neg' + str(neg_ratio) + 'epoch' + str(epoch)

with mlflow.start_run(run_name = run_name):

    # read data before model training
    train = pd.read_csv('./data/KKBOX/train_index_data.csv').values
    valid = pd.read_csv('./data/KKBOX/valid_index_data.csv').values
    test = pd.read_csv('./data/KKBOX/test_index_data.csv').values
    with open('./data/KKBOX/metadata.json') as f:
        metadata = json.load(f)

    # initialized TransE model object
    model = TransE(
        embedding_params={"embedding_size": emb_size},
        negative_ratio= neg_ratio,
        corrupt_side="h+t"
    )

    # MLflow log parameters
    mlflow.log_param('embedding_size',emb_size)
    mlflow.log_param('negative_ratio',neg_ratio)

    
    # train the model
    model.train(train_X=train, val_X=valid, metadata=metadata, epochs=epoch, batch_size=10000,
                early_stopping_rounds=10, restore_best_weight=False,
                optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                seed=12345, log_path=log_path, log_projector=False)

    # MLflow log artifact 1.training and validation loss curve
    train_loss = model.train_loss_history
    val_loss = model.val_loss_history
    epochs = range(1,len(train_loss)+1)
    plt.plot(epochs, train_loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig_fn = 'train_val_loss_{}_{}.png'.format('emb' + str(emb_size),'neg' + str(neg_ratio))
    plt.savefig(fig_fn)
    mlflow.log_artifact(fig_fn) # logging to mlflow
    plt.close()

    # # MLflow log artifact 2. model weights
    # model_weights = {'ent_emb': model.model_weights['ent_emb'].numpy().tolist(),
    #                     'rel_emb': model.model_weights['rel_emb'].numpy().tolist()}
    # with open("./data/KKBOX/model_weights.json", 'w') as f:
    #     json.dump(model_weights, f)
    # mlflow.log_artifact('./data/KKBOX/model_weights.json')

    # For Parameter Tuning -> use VALIDATION data
    # generate test data
    test_df = pd.read_csv('./data/KKBOX/test_data.csv')
    test_users, user_and_hasInterestItem  = generateTestData(test_df)

    # recommend and evaluate on TEST data
    test_users_rec_music = recommend(test_users)
    test_evaluate_result = evaluate(test_users_rec_music)

    # write in tensorboard log
    summary_writer = tf.summary.create_file_writer(log_path)
    with summary_writer.as_default():
        tf.summary.scalar('test-hit', test_evaluate_result['hit'], step=0)
        tf.summary.scalar('test-recall', test_evaluate_result['recall'], step=0)
        tf.summary.scalar('test-precision', test_evaluate_result['precision'], step=0)
        tf.summary.scalar('test-ndcg', test_evaluate_result['ndcg'], step=0)

    # MLflow log metrics
    mlflow.log_metric('test-hit',test_evaluate_result['hit'])
    mlflow.log_metric('test-recall',test_evaluate_result['recall'])
    mlflow.log_metric('test-precision',test_evaluate_result['precision'])
    mlflow.log_metric('test-ndcg',test_evaluate_result['ndcg']) 

