import pandas as pd
import numpy as np

book_rating = pd.read_csv('UbB_all.txt', header = None, sep = ' ', names = ['user_id', 'item_id','rating'])
print(len(book_rating))
book_rating = book_rating.sample(frac=1)

freq = book_rating.groupby('user_id').count().reset_index()[['user_id','item_id']].rename(columns = {'item_id':'Count'})

user_sub = freq[freq.Count>0].user_id.unique()
book_rating_sub = book_rating[book_rating.user_id.isin(user_sub)]

print(len(book_rating), len(user_sub))

# np.savetxt('UmM.txt', book_rating_sub,  delimiter = '/t', fmt='%d %d %.1f')