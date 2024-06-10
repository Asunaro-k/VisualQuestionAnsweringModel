from keras.layers import Input
from keras.models import Model


def VQAModel():
    visual_input = Input((3, 224, 224))
    text_token_input = Input((15, 512))
