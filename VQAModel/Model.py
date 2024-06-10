from keras.layers import Input
import keras.layers as layers
from keras.models import Model
from transformers import ViTImageProcessor, TFViTModel

def VQAModel():
    target_embedding_shape = (15, 512)
    visual_embedding_shape = (197, 768)
    text_embedding_shape = (15, 512)
    visual_embedding = Input(visual_embedding_shape)
    text_embedding = Input(text_embedding_shape)

    x_v = layers.Dense(
        units=target_embedding_shape[1], 
        activation='relu', 
        use_bias=True,
    )(visual_embedding)
    
    # x_v = layers.Reshape(
    #     (visual_embedding_shape[0], target_embedding_shape[1])
    # )(x_v)

    x_t = layers.Dense(
        units=target_embedding_shape[1], 
        activation='relu', 
        use_bias=True,
    )(text_embedding)
    
    # x_t = layers.Reshape(
    #     (text_embedding_shape[0], target_embedding_shape[1])
    # )(x_t)

    x = layers.Concatenate(axis=-1)([x_v, x_t])


    # some layers // todo

    # current shape (15, 512)
    return Model(inputs=[visual_embedding, text_embedding], output=x)

if __name__ == '__main__':
    import tensorflow
    from transformers import T5Tokenizer, TFT5Model
    from transformers import ViTImageProcessor, TFViTModel
    
    pretrained_t5_path = 'dump/t5-small'
    pretrained_vit_path = 'dump/vit-base-patch16-224-in21k'
    
    tokenizer = T5Tokenizer.from_pretrained(pretrained_t5_path)
    t5 = TFT5Model.from_pretrained(pretrained_t5_path)
    t5_encoder = t5.encoder

    vit = TFViTModel.from_pretrained(pretrained_vit_path)
    
    model = VQAModel()

    text_embedding = t5_encoder(
        tokenizer('I am a Ironman.', return_tensors='tf').input_ids
    )
    visual_embedding = vit(
        tensorflow.random.uniform((1, 3, 224, 224), minval=-1, maxval=1)
    ).last_hidden_state
    output = model(visual_embedding, text_embedding)

    print(output)
