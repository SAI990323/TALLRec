import pandas as pd
rating = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding="latin-1")
users = pd.read_csv('BX-Users.csv', sep=';', encoding="latin-1")
books = pd.read_csv('BX-Books.csv', sep=';', encoding="latin-1", error_bad_lines=False)
rating = pd.merge(rating, books, on='ISBN', how='inner')
books.to_csv('book_item_mapping.csv', index=True)
                
from tqdm import tqdm
user_dict = {}
item_id = {}
for index, row in tqdm(books.iterrows()):
    item_id[row['ISBN']] = index
for index, row in tqdm(rating.iterrows()):
    userid = row['User-ID']
    if not user_dict.__contains__(userid):
        user_dict[userid] = {
            'ISBN': [],
            'Book-Rating': [],
            'Book-Title': [],
            'Book-Author': [],
            'Year-Of-Publication': [],
        }
    user_dict[userid]['ISBN'].append(item_id[row['ISBN']])
    user_dict[userid]['Book-Rating'].append(float(row['Book-Rating']))
    user_dict[userid]['Book-Title'].append(row['Book-Title'])
    user_dict[userid]['Book-Author'].append(row['Book-Author'])
    user_dict[userid]['Year-Of-Publication'].append(row['Year-Of-Publication'])

new_user_dict = {}
for key in user_dict.keys():
    mx = max(mx, len(user_dict[key]['ISBN']))
    if len(user_dict[key]['ISBN'])  <= 3:
        pass
    else:
        new_user_dict[key] = user_dict[key]

import random
import json
user_list = list(new_user_dict.keys())
random.shuffle(user_list)
train_user = user_list[:int(len(user_list) * 0.8)]
valid_usser = user_list[int(len(user_list) * 0.8):int(len(user_list) * 0.9)]
test_user = user_list[int(len(user_list) * 0.9):]

def generate_csv(user_list, output_csv, output_json):
    nrows = []
    for user in user_list:
        item_id = user_dict[user]['ISBN']
        rating = [int(_ > 5) for _ in user_dict[user]['Book-Rating']]
        random.seed(42)
        random.shuffle(item_id)
        random.seed(42)
        random.shuffle(rating)
        nrows.append([user, item_id[:-1][:10], rating[:-1][:10], item_id[-1], rating[-1]])
    with open(output_csv, 'w') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(['user', 'history_item_id','history_rating','item_id','rating'])
        writer.writerows(nrows)
    Prompt_json = []
    for user in user_list:
        item_id = user_dict[user]['ISBN']
        rating = [int(_ > 5) for _ in user_dict[user]['Book-Rating']]
        book_title = user_dict[user]['Book-Title']
        book_author = user_dict[user]['Book-Author']
        random.seed(42)
        random.shuffle(item_id)
        random.seed(42)
        random.shuffle(rating)
        random.seed(42)
        random.shuffle(book_title)
        random.seed(42)
        random.shuffle(book_author)
        preference = []
        unpreference = []
        for i in range(min(len(item_id) - 1, 10)):
            if rating[i] == 1:
                preference.append("\"" + book_title[i] + "\"" + " written by " + book_author[i])
            else:
                unpreference.append("\"" + book_title[i] + "\"" + " written by " + book_author[i])
        preference_str = ""
        unpreference_str = ""
        for i in range(len(preference)):
            if i == 0:
                preference_str += preference[i]
            else:
                preference_str += ", " + preference[i]
        for i in range(len(unpreference)):
            if i == 0:
                unpreference_str += unpreference[i]
            else:
                unpreference_str += ", " + unpreference[i]
        target_preference_str = "Yes." if rating[-1] == 1 else "No."
        target_book_str = "\"" + book_title[-1] + "\"" + "written by" + book_author[-1]
        Prompt_json.append({
            "instruction": "Given the user's preference and unpreference, identify whether the user will like the target book by answering \"Yes.\" or \"No.\".",
            "input": f"User Preference: {preference_str}\nUser Unpreference: {unpreference_str}\nWhether the user will like the target book {target_book_str}?",
            "output": target_preference_str,
        })
    with open(output_json, 'w') as f:
        json.dump(Prompt_json, f, indent=4)

generate_csv(train_user, 'train_book.csv', 'train_book.json')
generate_csv(valid_usser, 'valid_book.csv', 'valid_book.json')
generate_csv(test_user, 'test_book.csv', 'test_book.json')