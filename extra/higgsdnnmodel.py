import tensorflow as tf
from tensorflow.keras import layers, Model

@tf.keras.utils.register_keras_serializable(package="HiggsAnalysis")
class HiggsClassifier(Model):
    def __init__(self, input_dim=24, **kwargs):
        super(HiggsClassifier, self).__init__(**kwargs)
        self.input_dim = input_dim
        
        self.dense1 = layers.Dense(64, activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.2)
        
        self.dense2 = layers.Dense(32, activation='relu')
        self.bn2 = layers.BatchNormalization()
        
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        
        return self.output_layer(x)