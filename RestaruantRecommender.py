import numpy as np
import pandas as pd
import ast
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

class RestaurantRecommender:
    def __init__(self, city_name):
        self.city_name = city_name
        self.model = KNeighborsClassifier(26)
    
    def load_file(self, file_path):
        businesses = pd.read_json(file_path, lines=True, orient='columns')
        city = businesses[(businesses['is_open'] == 1) & (businesses['city'] == self.city_name)]
        return city
    
    def preprocess_data(self, city_data):
        # helper functions
        def extract_keys(attr, key):
            if attr is None:
                return "{}"
            if key in attr:
                return attr.pop(key)
        def str_to_dict(attr):
            if attr is not None:
                return ast.literal_eval(attr)
            else:
                return ast.literal_eval("{}")

        city_data['BusinessParking'] = city_data.apply(lambda x: str_to_dict(extract_keys(x['attributes'], 'BusinessParking')), axis=1)
        city_data['Ambience'] = city_data.apply(lambda x: str_to_dict(extract_keys(x['attributes'], 'Ambience')), axis=1)
        city_data['GoodForMeal'] = city_data.apply(lambda x: str_to_dict(extract_keys(x['attributes'], 'GoodForMeal')), axis=1)
        city_data['Dietary'] = city_data.apply(lambda x: str_to_dict(extract_keys(x['attributes'], 'Dietary')), axis=1)
        city_data['Music'] = city_data.apply(lambda x: str_to_dict(extract_keys(x['attributes'], 'Music')), axis=1)

        df_attr = pd.concat([city_data['attributes'].apply(pd.Series), city_data['BusinessParking'].apply(pd.Series),
                             city_data['Ambience'].apply(pd.Series), city_data['GoodForMeal'].apply(pd.Series),
                             city_data['Dietary'].apply(pd.Series)], axis=1)
        df_attr_dummies = pd.get_dummies(df_attr)

        df_categories_dummies = pd.Series(city_data['categories']).str.get_dummies(',')

        result = city_data[['name', 'business_id', 'stars']]
        df_final = pd.concat([df_attr_dummies, df_categories_dummies, result], axis=1)
        df_final.drop('Restaurants', inplace=True, axis=1)

        mapper = {1.0: 1, 1.5: 2, 2.0: 2, 2.5: 3, 3.0: 3, 3.5: 4, 4.0: 4, 4.5: 5, 5.0: 5}
        df_final['stars'] = df_final['stars'].map(mapper)

        X = df_final.iloc[:, :-3]
        y = df_final['stars']
        return X, y, df_final
    
    def fit(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        self.model.fit(self.X_train, self.y_train)
    
    def save_model(self, filename):
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        self.model = joblib.load(filename)
    
    def recommend(self, df_final):
        test_set = df_final.iloc[-1:, :-3]
        X_val = df_final.iloc[:-1, :-3]
        y_val = df_final['stars'].iloc[:-1]
        
        self.model.fit(X_val, y_val)
        
        final_table = pd.DataFrame(self.model.kneighbors(test_set)[0][0], columns=['distance'])
        final_table['index'] = self.model.kneighbors(test_set)[1][0]
        final_table.set_index('index', inplace=True)
        
        result = final_table.join(df_final, on='index')
        return result[['distance', 'index', 'name', 'stars']].head(10)