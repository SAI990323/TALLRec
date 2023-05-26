import json
import pandas as pd
f = open('u.data', 'r')
data = f.readlines()
f = open('u.item', 'r', encoding='ISO-8859-1')
movies = f.readlines()
f = open('u.user', 'r')
users = f.readlines()

movie_names = [_.split('|')[1] for _ in movies] # movie_names[0] = 'Toy Story (1995)'
user_ids = [_.split('|')[0] for _ in users] # user_ids[0] = '1'
movie_ids = [_.split('|')[0] for _ in movies] # movie_ids[0] = '1'
interaction_dicts = dict()  
for line in data:
    user_id, movie_id, rating, timestamp = line.split('\t')
    if user_id not in interaction_dicts:
        interaction_dicts[user_id] = {
            'movie_id': [],
            'rating': [],
            'timestamp': [],
        }
    interaction_dicts[user_id]['movie_id'].append(movie_id)
    interaction_dicts[user_id]['rating'].append(int(int(rating) > 3))
    interaction_dicts[user_id]['timestamp'].append(timestamp)

with open('item_mapping.csv', 'w') as f:
    import csv
    writer = csv.writer(f)
    writer.writerow(['movie_id', 'movie_name'])
    for i, name in enumerate(movie_names):
        writer.writerow([i + 1, name])

sequential_interaction_list = []
seq_len = 10
for user_id in interaction_dicts:
    temp = zip(interaction_dicts[user_id]['movie_id'], interaction_dicts[user_id]['rating'], interaction_dicts[user_id]['timestamp'])
    temp = sorted(temp, key=lambda x: x[2])
    result = zip(*temp)
    interaction_dicts[user_id]['movie_id'], interaction_dicts[user_id]['rating'], interaction_dicts[user_id]['timestamp'] = [list(_) for _ in result]
    for i in range(10, len(interaction_dicts[user_id]['movie_id'])):
        sequential_interaction_list.append(
            [user_id, interaction_dicts[user_id]['movie_id'][i-seq_len:i], interaction_dicts[user_id]['rating'][i-seq_len:i], interaction_dicts[user_id]['movie_id'][i], interaction_dicts[user_id]['rating'][i], interaction_dicts[user_id]['timestamp'][i].strip('\n')]
        )
    
sequential_interaction_list = sequential_interaction_list[-10000:] # 10000 records


import csv
# save the csv file for baselines
with open('./data/train.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'history_movie_id', 'history_rating', 'movie_id', 'rating', 'timestamp'])
    writer.writerows(sequential_interaction_list[:int(len(sequential_interaction_list)*0.8)])
with open('./data/valid.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'history_movie_id', 'history_rating', 'movie_id', 'rating', 'timestamp'])
    writer.writerows(sequential_interaction_list[int(len(sequential_interaction_list)*0.8):int(len(sequential_interaction_list)*0.9)])
with open('./data/test.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'history_movie_id', 'history_rating', 'movie_id', 'rating', 'timestamp'])
    writer.writerows(sequential_interaction_list[int(len(sequential_interaction_list)*0.9):])

def csv_to_json(input_path, output_path):
    data = pd.read_csv(input_path)
    json_list = []
    for index, row in data.iterrows():
        row['history_movie_id'] = eval(row['history_movie_id'])
        row['history_rating'] = eval(row['history_rating'])
        L = len(row['history_movie_id'])
        preference = []
        unpreference = []
        for i in range(L):
            if int(row['history_rating'][i]) == 1:
                preference.append(movie_names[int(row['history_movie_id'][i]) - 1])
            else:
                unpreference.append(movie_names[int(row['history_movie_id'][i]) - 1])
        target_movie = movie_names[int(row['movie_id']) - 1]
        preference_str = ""
        unpreference_str = ""
        for i in range(len(preference)):
            if i == 0:
                preference_str += "\"" + preference[i] + "\""
            else:
                preference_str += ", \"" + preference[i] + "\""
        for i in range(len(unpreference)):
            if i == 0:
                unpreference_str += "\"" + unpreference[i] + "\""
            else:
                unpreference_str += ", \"" + unpreference[i] + "\""
        target_preference = int(row['rating'])
        target_movie_str = "\"" + target_movie + "\""
        target_preference_str = "Yes." if target_preference == 1 else "No."
        json_list.append({
            "instruction": "Given the user's preference and unpreference, identify whether the user will like the target movie by answering \"Yes.\" or \"No.\".",
            "input": f"User Preference: {preference_str}\nUser Unpreference: {unpreference_str}\nWhether the user will like the target movie {target_movie_str}?",
            "output": target_preference_str,
        })
        
    with open(output_path, 'w') as f:
        json.dump(json_list, f, indent=4)

# generate the json file for the TALLRec
csv_to_json('./data/train.csv', './data/train.json')
csv_to_json('./data/valid.csv', './data/valid.json')
csv_to_json('./data/test.csv', './data/test.json')