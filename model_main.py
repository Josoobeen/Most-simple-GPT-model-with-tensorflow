from model_parts.masking import 

def decoder_module(decoder_inputs, num_heads, head_dim, training, look_ahead_mask=None):
    #decoder_input, encoder_output, padding
    #decoder의 마스킹은 이후 정보에 대한 접근을 막아주는 역할
    #multi_head_attention 1 부분(self)
    multiheadout1 = multi_head_attention(decoder_inputs, decoder_inputs, decoder_inputs, 
                                         num_heads, head_dim, mask = look_ahead_mask)
    sub_out = sublayer_connection(decoder_inputs, multiheadout1, dropout = 0.2)
    sub_out = tf.keras.layers.LayerNormalization(epsilon=1e-9)(sub_out)
    
    #세번째 서브층 : 피드포워드 네트워크
    ffn_out = Dense((num_heads * head_dim)//2, activation = 'relu')(sub_out)
    ffn_out = Dense(num_heads * head_dim, activation = 'relu')(ffn_out)
    ffn_out = dropout2(ffn_out, training, dropout_rate = 0.2)
    ffn_out = tf.keras.layers.LayerNormalization(epsilon=1e-9)(sub_out + ffn_out)
    return ffn_out


def decoder(inputs, num_heads, head_dim, num_layers, training,look_ahead_mask=None):
    
    emb = Embedding(input_dim = vocab_size, output_dim = num_heads * head_dim)(inputs)
    emb *= tf.math.sqrt(tf.cast(num_heads * head_dim, tf.float32))
    emb = tf.math.add(positional_encoding(num_heads * head_dim, max_len), emb)
    outputs = Dropout(rate=0.2)(emb)
    
    for i in range(num_layers):
        outputs = decoder_module(outputs, num_heads, head_dim, training, look_ahead_mask=look_ahead_mask)

    return outputs
    
    
    
def GPT2(de_in, num_heads, head_dim, num_layers, training):
    de_look_mask = create_look_ahead_mask(de_in.shape[1])
    de_output = decoder(de_in, num_heads, head_dim, num_layers, training, 
                        look_ahead_mask=de_look_mask)

    out = Dense(vocab_size, activation = 'softmax')(de_output)

    return out
    
