import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_recommenders as tfrs

from recommender import Recommender
from user_embedding import UserModel
from item_embedding import ItemModel

# from utils import dataset_to_dataframe
from datetime import datetime

base_loc = r'D:\dev work\recommender systems\CARS'

train_ds = tf.data.Dataset.load("D:/dev work/recommender systems/CARS/data/ratings_train").cache() #data\ratings_train
test_ds = tf.data.Dataset.load("D:/dev work/recommender systems/CARS/data/ratings_test").cache()
ratings_all = tf.data.Dataset.load("D:/dev work/recommender systems/CARS/data/ratings_all").cache()

movie_titles = ratings_all.batch(1_000_000).map(lambda x: x["movie_title"])
user_ids = ratings_all.batch(1_000_000).map(lambda x: x["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

timestamps = np.concatenate(list(ratings_all.map(lambda x: x["timestamp"]).batch(100)))

max_timestamp = timestamps.max()
min_timestamp = timestamps.min()


timestamp_buckets = np.linspace(
    min_timestamp, max_timestamp, num=1000,
)

model = Recommender(
    use_timestamp = True,
    unique_user_ids = unique_user_ids,
    unique_movie_titles = unique_movie_titles,
    timestamps = timestamps,
    timestamp_buckets = timestamp_buckets
)

log_dir = os.path.join(base_loc ,"logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
# log_dir = "../../logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=0,
    embeddings_freq = 1,
    write_images = True)

model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

train_ds = train_ds.shuffle(100_000).batch(8192) #.cache()

model.fit(
    train_ds, 
    epochs=15, 
    verbose = 1,
    callbacks=[tensorboard_callback]
    )

#save model
base = r'D:\dev work\recommender systems\CARS\model_weights\{}'.format(datetime.now().strftime("%Y_%m_%d_%H_%M"))

if not os.path.exists(base):
    os.makedirs(base)

model_name = 'tf_rating_{}'.format(datetime.now().strftime("%Y_%m_%d_%H_%M"))
save_path = os.path.join(base,model_name)

model.save_weights(save_path)
