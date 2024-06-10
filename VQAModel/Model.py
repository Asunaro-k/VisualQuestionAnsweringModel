from keras.layers import Input
from keras.models import Model
from transformers import ViTImageProcessor, TFViTModel


def VQAModel(pretrained_vit_path='dump/vit-base-patch16-224-in21k'):
    visual_input = Input((3, 224, 224))
    text_token_input = Input((15, 512))
    # processor = ViTImageProcessor.from_pretrained(pretrained_vit_path)
    vit = TFViTModel.from_pretrained(pretrained_vit_path)
    visual_embedding = vit(visual_input).last_hidden_state
    # current shape is  (batch, 197, 768)
    

