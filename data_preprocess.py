import pandas as pd
import json

# function to pair 2 col into a relation -> (h, h-type, r, t, t-type)
def hrt_pair(df, h_col, t_col, r, h_col_type, t_col_type):
    ''' 
    Parameters
    ----------
        df: from dataframe
        h_col: head column
        t_col: tail column
        r: relation between h_col and t_col
        h_col_type: type of head column
        t_col_type: type of tail column
        
    Returns
    -------
        hrt_df: (h, h-type, r, t, t-type)
    '''
    hrt_df = pd.concat([df[h_col], df[t_col]], axis=1)
    hrt_df['r'] = r
    hrt_df['h-type'] = h_col_type
    hrt_df['t-type'] = t_col_type
    hrt_df = hrt_df.rename(columns={h_col:'h',t_col:'t'})
    hrt_df = hrt_df[['h','h-type','r','t','t-type']]
    hrt_df = hrt_df.dropna()

    return hrt_df


#【train dataframe】

# read train data
train_df = pd.read_csv('./data/KKBOX/train.csv')
# drop column
train_df = train_df.drop(columns=['source_system_tab', 'source_screen_name', 'source_type'])
# drop row where target=0
train_df = train_df[train_df.target != 0]
# relation
train_df['target'] = train_df['target'].replace({1:'has_interest'})
# rename column
train_df = train_df.rename(columns={'msno':'h','target':'r','song_id':'t'})
train_df['h-type'] = 'member'
train_df['t-type'] = 'song'
# add prefix (because there are id belongs to both user and song)
train_df['h'] = 'uid_' + train_df['h']
train_df['t'] = 'sid_' + train_df['t']
# change column order
train_df = train_df[['h','h-type','r','t','t-type']]


# 【song dataframe】

# manually parse
f = open('./data/KKBOX/songs.csv',encoding='utf-8'); next(f)
song_manual = []
for l in f:
    song_manual.append(l.split(','))

# generate dataframe
song_df = pd.DataFrame(song_manual, columns=['song_id','song_length','genre_ids','artist_name','composer','lyricist','language','none'])
song_df['language'] = song_df['language'].str.rstrip('\n')
song_df = song_df.loc[:,'song_id':'language']
song_df['song_length'] = song_df['song_length'].astype(int)
# add prefix
song_df['song_id'] = 'sid_' + song_df['song_id']

# (1) song_length
# to htr_pair
song_length_df = hrt_pair(song_df, 'song_id','song_length', 'length', 'song', 'song_length')
# ms -> m
song_length_df['t'] /= 1000
# category: >3min & <3 min
song_length_df['t'] = ['>=3min' if length >= 180 else '<3min' for length in song_length_df['t']]

# (2) song genre
# to htr_pair
song_genre_df = hrt_pair(song_df, 'song_id', 'genre_ids', 'genre', 'song', 'song_genre')
# deal with multiple genre (split + explode)
song_genre_df = song_genre_df.assign(t = song_genre_df['t'].str.split('|')).explode('t', ignore_index=True)
# add prefix to genre
song_genre_df['t'] = 'genre_' + song_genre_df['t']

# (3) song artist
# to htr_pair
song_artist_df = hrt_pair(song_df, 'song_id', 'artist_name', 'artist', 'song', 'song_artist')

# (4) song language
# to htr_pair
song_language_df = hrt_pair(song_df, 'song_id', 'language', 'language', 'song', 'song_language')
# add prefix to language
song_language_df['t'] = 'language_' + song_language_df['t'].astype('string')


# 【member dataframe】

# read member data
member_df = pd.read_csv('./data/KKBOX/members.csv')
# add prefix
member_df['msno'] = 'uid_' + member_df['msno']

# (1) member city
# to htr_pair
member_city_df = hrt_pair(member_df, 'msno', 'city', 'city', 'member', 'member_city')
# add prefix to city
member_city_df['t'] = 'city_' + member_city_df['t'].astype('string')

# (2) member age
# to htr_pair
member_age_df = hrt_pair(member_df, 'msno', 'bd', 'age', 'member', 'member_age')
# delete outlier(age=0 & age<0 & age>80)
member_age_df.loc[member_age_df['t'] < 0, 't'] = 0
member_age_df.loc[member_age_df['t'] > 80, 't'] = 0
member_age_df = member_age_df.drop(member_age_df[member_age_df['t']==0].index)
# slice bucket
bins = [0, 20, 30, 40, 50, 80]
member_age_df['t'] = pd.cut(member_age_df['t'], bins,labels=['<20','20-30','30-40','40-50','>=50'])

# (3) member gender
# to htr_pair
member_gender_df = hrt_pair(member_df, 'msno', 'gender', 'gender', 'member', 'member_gender')

# (4) member duration
# count year
member_df['member_years'] = member_df['expiration_date']//10000 - member_df['registration_init_time']//10000
# to htr_pair
member_duration_df = hrt_pair(member_df, 'msno', 'member_years', 'member_duration_in_year', 'member', 'member_duration_in_year')
# delete row where t<0
member_duration_df = member_duration_df.drop(member_duration_df[member_duration_df['t']<0].index)
# add prefix to duration
member_duration_df['t'] = 'year_' + member_duration_df['t'].astype('string')


# output csv

interest_df = train_df[['h','r','t']]
other_df = pd.concat([song_length_df, song_genre_df,\
                 song_artist_df, song_language_df, member_city_df,\
                member_age_df, member_gender_df, member_duration_df], ignore_index=True)[['h','r','t']]
kg_df = pd.concat([interest_df, other_df], ignore_index=True)
interest_df.to_csv('./data/KKBOX/kgdata_interest.csv', index=False)
other_df.to_csv('./data/KKBOX/kgdata_other.csv', index=False)
kg_df.to_csv('./data/KKBOX/kgdata_all.csv', index=False)

# output type dictionary 

# all kg with type
kg_type_df = pd.concat([train_df, song_length_df, song_genre_df,\
                 song_artist_df, song_language_df, member_city_df,\
                member_age_df, member_gender_df, member_duration_df], ignore_index=True)
# entity type dictionary for head
unique_h_df = kg_type_df.drop_duplicates(subset=['h'])
h_dict = dict(zip(unique_h_df['h'],unique_h_df['h-type']))
# entity type dictionary for tail
unique_t_df = kg_type_df.drop_duplicates(subset=['t'])
t_dict = dict(zip(unique_t_df['t'],unique_t_df['t-type']))
# merge 
h_dict.update(t_dict)
type_dict = h_dict
with open('./data/KKBOX/type_dict.json', 'w') as f:
    json.dump(type_dict, f)

# create entity groupby type dict
entity_groupby_type = {type:[entity for entity in type_dict.keys() if type_dict[entity] == type] for type in set(type_dict.values())}
with open('./data/KKBOX/entity_groupby_type.json', 'w') as f:
    json.dump(entity_groupby_type, f)