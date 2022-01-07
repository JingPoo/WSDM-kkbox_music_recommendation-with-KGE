import json
import numpy as np
import pandas as pd
import tensorflow as tf
from KGE.data_utils import index_kg, convert_kg_to_index
from KGE.models.translating_based.TransE import TransE
from sklearn.model_selection import train_test_split


def kg_train_val_test_split(data, test_size, val_size):
    '''
        Parameters
        ----------
            data: the data to be split
            test_size: if float, represent the proportion of the dataset to include in the test split
                    if int, represents the absolute number of test samples
            val_size: same as test_size, but is the proportion of the train data

        Returns
        -------
            train, valid, test
    '''
    # sort by first column
    sorted_data = data[np.argsort(data[:,0])]
    # get unique user index and count
    unique_head, index, count = np.unique(sorted_data[:,0], return_index=True, return_counts=True)

    # train test split
    train_origin = [] 
    test = [] 
    for i in range(len(index)):
        if count[i]>1:
            tr, te = train_test_split(sorted_data[index[i]:index[i] + count[i], :], test_size=test_size, random_state=i)
            train_origin.append(tr)
            test.append(te)
        else: #只有一筆
            test.append(sorted_data[index[i],:])
            
    # train val split
    valid = []
    train = []
    for j in range(len(train_origin)):
        if len(train_origin[j])>1:
            tr, va = train_test_split(train_origin[j], test_size=val_size, random_state=j)
            train.append(tr)
            valid.append(va)
        else: #只有一筆
            valid.append(train_origin[j])
     
    return np.vstack(train), np.vstack(valid), np.vstack(test)


# load data
interest_data = pd.read_csv('./data/KKBOX/kgdata_interest.csv').to_numpy()
# train(train 0.9: val 0.1) 0.67 : test 0.33
train, valid, test = kg_train_val_test_split(interest_data, 0.33, 0.1)
# read other data
other_data = pd.read_csv('./data/KKBOX/kgdata_other.csv').to_numpy()
# concate kgdata_interest & kgdata_other as train
train = np.concatenate((train, other_data))

# output data before index
pd.DataFrame(train,columns=['h','r','t']).to_csv('./data/KKBOX/train_data.csv', index=False)
pd.DataFrame(valid,columns=['h','r','t']).to_csv('./data/KKBOX/valid_data.csv', index=False)
pd.DataFrame(test,columns=['h','r','t']).to_csv('./data/KKBOX/test_data.csv', index=False)

# index the kg data
metadata = index_kg(train)

# output metadata json
with open('./data/KKBOX/metadata.json', 'w') as f:
    json.dump(metadata, f)

# convert kg into index
train = convert_kg_to_index(train, metadata["ent2ind"], metadata["rel2ind"])
valid = convert_kg_to_index(valid, metadata["ent2ind"], metadata["rel2ind"])
test = convert_kg_to_index(test, metadata["ent2ind"], metadata["rel2ind"])

# output data after index
pd.DataFrame(train,columns=['h','r','t']).to_csv('./data/KKBOX/train_index_data.csv', index=False)
pd.DataFrame(valid,columns=['h','r','t']).to_csv('./data/KKBOX/valid_index_data.csv', index=False)
pd.DataFrame(test,columns=['h','r','t']).to_csv('./data/KKBOX/test_index_data.csv', index=False)