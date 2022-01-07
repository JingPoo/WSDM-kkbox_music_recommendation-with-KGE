import json
import pandas as pd

def show_kg_summary():
  
    kg = pd.read_csv('./data/KKBOX/kgdata_all.csv')
    with open('./data/KKBOX/type_dict.json') as f:
        type_dict = json.load(f)

    print('Summary of KG\n'
          '-------------')
    # - number of hrt triplets
    print('number of hrt triplets: ', kg.shape[0])
    # - number of entities
    # make a dataframe append h & t -> calculate N_entities
    ht_df = kg['h'].append(kg['t'],ignore_index=True)
    print('number of entities: ', ht_df.nunique())
    # - number of entity type(in dict)
    print('number of entity type: ', len(list(set(type_dict.values()))))
    # - number of entities group by type(in dict)
    entity_groupby_type = {type:[entity for entity in type_dict.keys() if type_dict[entity] == type] for type in set(type_dict.values())}
    N_entity_groupby_type = {type:len(entity_groupby_type[type]) for type in entity_groupby_type.keys()}
    print('number of entities group by type: ', N_entity_groupby_type)
    # - number of relations
    print('number of relations: ', kg['r'].nunique())


def show_data_summary():
      
    train_df = pd.read_csv('./data/KKBOX/train_data.csv')
    valid_df = pd.read_csv('./data/KKBOX/valid_data.csv')
    test_df = pd.read_csv('./data/KKBOX/test_data.csv')  
    with open('./data/KKBOX/type_dict.json') as f:
        type_dict = json.load(f)
     
    print('Summary of train, validation, test data\n'
        '---------------------------------------')
    # - number of hrt triplets
    N_hrt_triple = {'train':len(train_df), 'validation':len(valid_df), 'test':len(test_df)}
    print('number of hrt triplets: ', N_hrt_triple)  
    # - number of has_interest hrt triplet
    N_interest_hrt_triple = {'train':train_df[train_df['r'] == 'has_interest'].shape[0],\
                            'validation':valid_df[valid_df['r'] == 'has_interest'].shape[0],\
                            'test':test_df[test_df['r'] == 'has_interest'].shape[0]}
    print('number of has_interest hrt triplets: ', N_interest_hrt_triple) 
    
    ht_train_df = train_df['h'].append(train_df['t'],ignore_index=True)
    type_ht_train_df = [type_dict.get(ent) for ent in ht_train_df.unique()]
    ht_valid_df = valid_df['h'].append(valid_df['t'],ignore_index=True)
    type_ht_valid_df = [type_dict.get(ent) for ent in ht_valid_df.unique()]
    ht_test_df = test_df['h'].append(test_df['t'],ignore_index=True)
    type_ht_test_df = [type_dict.get(ent) for ent in ht_test_df.unique()]
    
    # - number of distinct user
    N_user = {'train':type_ht_train_df.count('member'), 'validation':type_ht_valid_df.count('member'),\
            'test':type_ht_test_df.count('member')}
    print('number of distinct user: ', N_user)
    # - number of distinct item
    N_item = {'train':type_ht_train_df.count('song'), 'validation':type_ht_valid_df.count('song'),\
            'test':type_ht_test_df.count('song')}
    print('number of distinct item: ', N_item)


show_kg_summary()
print()
show_data_summary()