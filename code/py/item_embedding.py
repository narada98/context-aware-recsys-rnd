import tensorflow as tf 

'''
this handles embedding item Identifiers and contextual data.
movie title itself is used as the contexual information here.
using timestamp is 
'''

class ItemModel(tf.keras.Model):
    def __init__(
        self,
        unique_movie_titles,
        ):
        super().__init__()

        self.max_tokens = 10000
        self.unique_movie_titles = unique_movie_titles

        self.embed_item_id = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary = self.unique_movie_titles,
                mask_token =None
            ),
            tf.keras.layers.Embedding(
                input_dim = len(self.unique_movie_titles)+1,
                output_dim = 32
            )
        ])


        self.textvectorizer = tf.keras.layers.TextVectorization(
            max_tokens = self.max_tokens
        )

        self.embed_item_title = tf.keras.Sequential([
            self.textvectorizer,

            tf.keras.layers.Embedding(
                input_dim = self.max_tokens,
                output_dim = 32,
                mask_zero = True
            ),

            tf.keras.layers.GlobalAveragePooling1D() # reduces dimensionality to 1d (embedding layer embeddeds each word in a title one by one)
        ])

        self.textvectorizer.adapt(self.unique_movie_titles)
    
    def call(self, inputs):

        movie_title = inputs

        return tf.concat([
            self.embed_item_id(movie_title),
            self.embed_item_title(movie_title)
        ],
        axis = 1)
        
        # return self.embed_item_title(inputs['movie_title'])