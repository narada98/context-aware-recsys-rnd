import numpy as np
import tensorflow as tf

from recommender import Recommender

test_ds = tf.data.Dataset.load("D:/dev work/recommender systems/CARS/data/ratings_test")
ratings_all = tf.data.Dataset.load("D:/dev work/recommender systems/CARS/data/ratings_all")

loaded_model = Recommender(
    use_timestamp = True,
    unique_user_ids = unique_user_ids,
    unique_movie_titles = unique_movie_titles,
    timestamps = timestamps,
    timestamp_buckets = timestamp_buckets
)

loaded_model.load_weights(r"..\..\models\tf_rating_2024_04_02_15_01")
loaded_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

loaded_model.evaluate(test_ds.shuffle(100_000).batch(1024), return_dict=True)

