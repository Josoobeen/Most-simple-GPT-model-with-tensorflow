import tensorflow as tf
import keras

def scaled_dot_product_attention(q, k, v, mask = None):
    matmul_qk = tf.matmul(q, k, transpose_b = True)# 뒷부분을 반전시킴
    depth = tf.cast(tf.shape(k)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)
    
    #shape은 k[0], k[1], k[1]로 바뀜
    # dim이 바뀐 key와 바뀌지 않은 q를 행렬곱하고 k_dim_size의 제곱근으로 나눠줌
    
  
    #masking은 어텐션 스코어 행렬에 마스킹 할 부분에 매우 작은 음수값을 넣는다.
    #어차피 소프트 맥스 함수 지나면 0이 된다.
    #padding을 어텐션에 집어넣지 않기 위함
    #mask = (batch_size, 1, 1, key의 문장길이)
    if mask is not None:
        logits += (mask * -1e9)
    
    attention_weights = keras.layers.Softmax()(logits)
    
    output = tf.matmul(attention_weights, v)
    return output, attention_weights
