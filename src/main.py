from restaurant_recommender import RestaurantRecommender
import json

recommender = RestaurantRecommender()

# training dataset
processed_training_dataset = recommender.preprocess_training_dataset("data/yelp_academic_dataset_business.json", 'Vancouver')
recommender.save_processed_training_dataset(processed_training_dataset, "test/csv/PROCESSED_TRAINING_DATASET_A.csv")
recommender.load_processed_training_dataset("test/csv/PROCESSED_TRAINING_DATASET_A.csv")

# processing target
with open('test/json/chick_asta_grill.json', 'r') as json_file:
    data = json.load(json_file)
preprocess_target = recommender.preprocess_target(data)
preprocess_target.to_csv('test/csv/PROCESSED_TARGET.csv', index=False)

# training model
model = recommender.train_model()
recommender.save_model(model, "models/VANCOUVER_MODEL_A")
recommender.load_model("models/VANCOUVER_MODEL_A")

# recommendation
recommendation = recommender.recommend(preprocess_target)

print(recommendation)