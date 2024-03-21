import scaled_dot_product_attention.

def multi_head_attention(q, k, v, num_heads, head_dim, mask = None):
    batch_size = tf.shape(q)[0]
    length = tf.shape(q)[1]
    q = Dense(num_heads * head_dim, use_bias = False)(q)
    k = Dense(num_heads * head_dim, use_bias = False)(k)
    v = Dense(num_heads * head_dim, use_bias = False)(v)

    q = tf.transpose(tf.concat(tf.split(q[:,:,tf.newaxis,:], head_dim, axis = -1), axis = 2), perm = [0, 2, 1, 3])
    k = tf.transpose(tf.concat(tf.split(k[:,:,tf.newaxis,:], head_dim, axis = -1), axis = 2), perm = [0, 2, 1, 3])
    v = tf.transpose(tf.concat(tf.split(v[:,:,tf.newaxis,:], head_dim, axis = -1), axis = 2), perm = [0, 2, 1, 3])


    scaled_attention_out, weights = scaled_dot_product_attention(q, k, v, mask = mask)

    scaled_attention_out = tf.transpose(scaled_attention_out, perm = [0,2,1,3])
    scaled_attention_out = tf.reshape(scaled_attention_out, (batch_size,length,num_heads * head_dim))
    outputs = Dense(num_heads * head_dim, use_bias = False)(scaled_attention_out)
  
    return outputs
