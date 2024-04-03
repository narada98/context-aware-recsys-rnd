import tensorflow as tf

from user_embedding import UserModel
from item_embedding import ItemModel


class RatingModel(tf.keras.Model):

  def __init__(

    self,
    # user_model,
    # item_model
    use_timestamp,
    unique_user_ids, 
    timestamps, 
    timestamp_buckets,
    unique_movie_titles
    ):
    
    super().__init__()

    embedding_dimension = 32
    self.use_timestamp = use_timestamp
    self.unique_user_ids = unique_user_ids
    self.timestamps = timestamps
    self.timestamp_buckets = timestamp_buckets
    self.unique_movie_titles = unique_movie_titles
    # self.user_model = user_model
    # self.item_model = item_model

    self.user_model = UserModel(
      use_timestamp = self.use_timestamp,
      unique_user_ids = self.unique_user_ids, 
      timestamps = self.timestamps, 
      timestamp_buckets = self.timestamp_buckets
      )

    self.item_model = ItemModel(
      unique_movie_titles = self.unique_movie_titles
      )

    # Compute predictions.
    self.ratings = tf.keras.Sequential([
      # Learn multiple dense layers.
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(64, activation="relu"),
      # Make rating predictions in the final layer.
      tf.keras.layers.Dense(1)
  ])

  def call(self, inputs):

    user_id, timestamp, movie_title = inputs

    user_embedding = self.user_model((user_id,timestamp))
    movie_embedding = self.item_model(movie_title)

    return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))