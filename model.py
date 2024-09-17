import numpy as np
import pandas as pd
import ast
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

businesses = pd.read_json("data/yelp_academic_dataset_business.json", lines=True, orient='columns')

city = businesses[(businesses['is_open'] == 1) & (businesses['city'] == 'Vancouver')]

vancouver = city[['business_id','name','address', 'categories', 'attributes','stars']]

rest = vancouver[vancouver['categories'].str.contains('Restaurant.*')==True].reset_index()

def extract_keys(attr, key):
    if attr == None:
        return "{}"
    if key in attr:
        return attr.pop(key)
    
def str_to_dict(attr):
    if attr != None:
        return ast.literal_eval(attr)
    else:
        return ast.literal_eval("{}")  

rest['BusinessParking'] = rest.apply(lambda x: str_to_dict(extract_keys(x['attributes'], 'BusinessParking')), axis=1)
rest['Ambience'] = rest.apply(lambda x: str_to_dict(extract_keys(x['attributes'], 'Ambience')), axis=1)
rest['GoodForMeal'] = rest.apply(lambda x: str_to_dict(extract_keys(x['attributes'], 'GoodForMeal')), axis=1)
rest['Dietary'] = rest.apply(lambda x: str_to_dict(extract_keys(x['attributes'], 'Dietary')), axis=1)
rest['Music'] = rest.apply(lambda x: str_to_dict(extract_keys(x['attributes'], 'Music')), axis=1)

df_attr = pd.concat([ rest['attributes'].apply(pd.Series), rest['BusinessParking'].apply(pd.Series),
                    rest['Ambience'].apply(pd.Series), rest['GoodForMeal'].apply(pd.Series), 
                    rest['Dietary'].apply(pd.Series) ], axis=1)
df_attr_dummies = pd.get_dummies(df_attr)

df_categories_dummies = pd.Series(rest['categories']).str.get_dummies(',')

result = rest[['name','business_id','stars']]

df_final = pd.concat([df_attr_dummies, df_categories_dummies, result], axis=1)
df_final.drop('Restaurants',inplace=True,axis=1)

mapper = {1.0:1,1.5:2, 2.0:2, 2.5:3, 3.0:3, 3.5:4, 4.0:4, 4.5:5, 5.0:5}
df_final['stars'] = df_final['stars'].map(mapper)

X = df_final.iloc[:,:-3]
y = df_final['stars']

X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X, y, test_size=0.3, random_state=1)

# Define the parameter grid

knn = KNeighborsClassifier(n_neighbors=25)
knn.fit(X_train_knn, y_train_knn)

# Print the best parameters and the best score


print(df_final.iloc[-2, -3:])

test_set =  df_final.iloc[:1,:-3]

X_val =  df_final.iloc[1:,:-3]
y_val = df_final['stars'].iloc[:-1]

n_knn = knn.fit(X_val, y_val)

final_table = pd.DataFrame(n_knn.kneighbors(test_set)[0][0], columns = ['distance'])
final_table['index'] = n_knn.kneighbors(test_set)[1][0]
final_table.set_index('index')


# get names of the restaurant that similar to the "Steak & Cheese & Quick Pita Restaurant"
result = final_table.join(df_final,on='index')
print(result[['distance','index','name','stars']].head(10))