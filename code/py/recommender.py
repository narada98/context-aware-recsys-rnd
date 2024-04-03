import tensorflow as tf
import tensorflow_recommenders as tfrs

from typing import Dict, Text

from rating_model import RatingModel

class Recommender(tfrs.models.Model):

  def __init__(
    self,
    use_timestamp,
    unique_user_ids,
    unique_movie_titles,
    timestamps,
    timestamp_buckets
    ):

    super().__init__()

    self.use_timestamp = use_timestamp
    self.unique_user_ids = unique_user_ids
    self.timestamps = timestamps
    self.timestamp_buckets = timestamp_buckets
    self.unique_movie_titles = unique_movie_titles

    self.rating_model: tf.keras.Model = RatingModel(

        use_timestamp = self.use_timestamp,
        unique_user_ids = self.unique_user_ids, 
        timestamps = self.timestamps, 
        timestamp_buckets = self.timestamp_buckets,
        unique_movie_titles = self.unique_movie_titles
      )

    self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
      loss = tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

  def call(self, inputs):
    return self.rating_model(inputs)

  def compute_loss(self, features, training=False):
    
    user_id, rating, movie_title, timestamp = features['user_id'], features['user_rating'], features['movie_title'], features['timestamp']
    rating_predictions = self((user_id, timestamp, movie_title))

    # The task computes the loss and the metrics.
    return self.task(labels=rating, predictions=rating_predictions)