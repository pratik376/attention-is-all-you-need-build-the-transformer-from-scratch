"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
from collections import defaultdict
import json
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):

    dictonary= {}
    count=-1

    for index, word in enumerate(specials):
        count+=1
        dictonary[word]=index
    
    for sentence in sentences:

        for word in sentence.split(' '):

            
            if not word in dictonary:
                count+=1
                dictonary[word]=count
    
    # dictonary= json.dumps(dictonary)
    return dictonary
    



    

    
    


  

    # TODO: build a token-to-id dict with specials first, then corpus tokens in first-seen order.

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    # TODO: build the inverse id-to-token dictionary from token_to_id

    id_to_token= {}

    for key, value in token_to_id.items():

        id_to_token[value]=key
    
    return id_to_token

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    # TODO: convert whitespace tokens of `sentence` to ids via `token_to_id`, using `unk_token`'s id for OOV
    encoding=[]
    if not sentence:
        return []

    for word in sentence.split(' '):
        
        if not word in token_to_id:
            encoding.append(token_to_id[unk_token])
        else:
            encoding.append(token_to_id[word])
    
    return encoding

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    # TODO: map each id in ids to its token string via id_to_token and return the list
    
    tokens=[]

    for idx in ids:

        tokens.append(id_to_token[idx])
    
    return tokens

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    # TODO: return a list of length exactly max_len, padding with pad_id or truncating.

    if len(ids) > max_len:
        return ids[:max_len]
    
    else:

        for i in range(max_len- len(ids)):
            ids.append(pad_id)
    
    return ids

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    # TODO: stack padded id sequences into a (B, L) torch.long tensor
    
    sentence= torch.tensor(padded_sequences, dtype=torch.long)
    return sentence

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    # TODO: rescale embeddings by sqrt(d_model) as in the original Transformer paper
    
    return math.sqrt(d_model) * embeddings

# Step 8 - compute_positional_div_term
import torch
import math

def compute_positional_div_term(d_model):
    # TODO: return a 1D FloatTensor of length d_model // 2 holding the sinusoidal frequency divisors
    answer=torch.empty(d_model//2, dtype=torch.float)

    for i in range(d_model//2):

        term= math.exp( 2 * i  * (-math.log(10000)/d_model))
        answer[i]= term
    
    return answer

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    # TODO: build a column vector of position indices from 0 to max_len-1

    return torch.arange(max_len,dtype=torch.float32).reshape(-1,1)

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    # TODO: write sin(position * div_term) into the even-indexed columns of pe and return it

    sin_postions= torch.sin(position * div_term)

    pe[:, 0::2]= sin_postions

    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    # TODO: fill the odd-indexed columns of pe with cos(position * div_term)
    
    cos_position= torch.cos(position* div_term)

    pe[:,1::2]= cos_position

    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    # TODO: build the (max_len, d_model) sinusoidal positional encoding matrix

    positional_matrix=torch.zeros(max_len,d_model)
    position = build_position_index_column(max_len)
    term=compute_positional_div_term(d_model)

    positional_matrix[:,0::2]= torch.sin(position * term)
    positional_matrix[:,1::2]= torch.cos(position * term)

    return positional_matrix

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    # TODO: add the first L rows of positional_encoding to embedded_batch and return the sum.
    
    B,L,d_model= embedded_batch.shape

    return embedded_batch + positional_encoding[:L]

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    # TODO: build a boolean mask marking non-pad positions, shaped for broadcasting against attention scores

    B,L =token_ids.shape

    return (token_ids != pad_id).reshape(B,1,1,L)

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    # TODO: build a lower-triangular boolean causal mask of shape (1, 1, seq_len, seq_len)
    mask= torch.tril(torch.ones(seq_len,seq_len)) ==1

    return mask.reshape(1,1,seq_len,seq_len)

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    # TODO: combine a (B,1,1,L) padding mask with a (1,1,L,L) causal mask into (B,1,L,L).
    
    return padding_mask & causal_mask

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    # TODO: matmul query with the transpose of key over the last two axes
    
    return query @ key.transpose(-2,-1)

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    # TODO: divide raw attention scores by sqrt(d_k) to stabilize softmax inputs
    
    return scores/ math.sqrt(d_k)

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    mask = mask.to(torch.bool)
    return torch.where(mask, scores, torch.full_like(scores, float("-inf")))

# Step 20 - softmax_attention_weights
import torch

def softmax_attention_weights(masked_scores):
    weights = torch.softmax(masked_scores, dim=-1)
    all_masked = torch.isneginf(masked_scores).all(dim=-1, keepdim=True)
    return torch.where(all_masked, torch.zeros_like(weights), weights)

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    # TODO: combine attention weights (..., Lq, Lk) with value (..., Lk, d_v)
    
    return attention_weights @ value

# Step 22 - scaled_dot_product_attention
import torch
import math

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    
    # 1) raw attention scores: Q @ K^T
    row_score = compute_raw_attention_scores(query, key)
    
    # 2) scale by sqrt(d_k)
    d_k = query.shape[-1]
    scaled_attention = scale_attention_scores(row_score, d_k)
    
    # 3) optionally apply mask
    if mask is not None:
        masked_scores = mask_attention_scores_with_neg_inf(scaled_attention, mask)
    else:
        masked_scores = scaled_attention
    
    # 4) softmax over the last axis
    attention_weights = softmax_attention_weights(masked_scores)
    
    # 5) weighted sum of values
    context = apply_attention_weights_to_values(attention_weights, value)
    
    return context, attention_weights

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    # TODO: reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)
    
    B,L,d_model=tensor.shape

    return tensor.reshape(B,L,num_heads, d_model//num_heads)

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    # TODO: rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    B, L, num_heads, d_k = split_tensor.shape

    return split_tensor.transpose(1,2)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    # TODO: merge the head axis back into the feature axis to reconstruct d_model
    B,H,L,DK= multi_head_tensor.shape

    return multi_head_tensor.transpose(1,2).reshape(B,L,H*DK)

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    # TODO: return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    
    output= x @ weight.T

    if bias is None:
        return output
    else:
        output= output +bias
        return output

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    # TODO: project x into separate query, key, and value tensors via three linear layers


    Q=x @ w_q.T 
    if b_q is not None:
        Q= Q +b_q

    K= x @ w_k.T 
    if b_k is not None:
        K= K + b_k
    
    V= x @ w_v.T 
    if b_v is not None:
        V= V + b_v


    return Q,K,V

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    # TODO: split each of q, k, v into (B, num_heads, L, d_k) and return as a tuple
    
    B,L, d_model= q.shape
    d_k= d_model // num_heads

    q= q.reshape(B,L,num_heads,d_k).transpose(1,2)
    k= k.reshape(B,L,num_heads,d_k).transpose(1,2)
    v= v.reshape(B,L,num_heads,d_k).transpose(1,2)

    return (q,k,v)

# Step 29 - multi_head_scaled_dot_product_attention
def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    # Run scaled dot-product attention over already-split multi-head tensors
    context, attention_weights = scaled_dot_product_attention(q_h, k_h, v_h, mask)
    return context, attention_weights

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    # TODO: merge the head axis back into d_model and apply the output linear projection.
    B,H,L,dk= context.shape

    context= context.transpose(1,2).reshape(B,L,H *dk)

    answer=apply_linear_projection(context,w_o, b_o)

    return answer

# Step 31 - assemble_multi_head_attention_forward
def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    # Project separately
    q = apply_linear_projection(query, w_q, None)
    k = apply_linear_projection(key, w_k, None)
    v = apply_linear_projection(value, w_v, None)

    # Split each one independently
    q_h = transpose_heads_before_sequence(split_last_dim_into_heads(q, num_heads))
    k_h = transpose_heads_before_sequence(split_last_dim_into_heads(k, num_heads))
    v_h = transpose_heads_before_sequence(split_last_dim_into_heads(v, num_heads))

    # Attention per head
    context_h, _ = multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask)

    # Merge heads back
    merged = merge_heads_back_to_model_dim(context_h)

    # Output projection
    out = apply_linear_projection(merged, w_o, None)

    return out

# Step 32 - apply_ffn_first_linear_and_relu
import torch

def apply_ffn_first_linear_and_relu(x, w1, b1):
    # TODO: project x by w1, add b1, then apply a ReLU activation.
    
    z= x @ w1  + b1 

    return torch.relu(z)

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    # TODO: project hidden (..., d_ff) back to (..., d_model) via w2 and b2.
    
    output = hidden @ w2 

    if b2 is not None:
        output += b2

    return output

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    # TODO: compose the two FFN linears with a ReLU in between, returning shape (B, T, d_model).

    output=apply_ffn_first_linear_and_relu(x, w1, b1)
    output= apply_ffn_second_linear(output,w2,b2)

    return output

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    # TODO: return (mean, variance) reduced over the last dim with shape (..., 1)
    
    mean=x.mean(dim=-1,keepdims=True)
    variance= x.var(dim=-1, keepdims=True,unbiased=False)

    return (mean,variance)

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    # TODO: standardize x along the last axis then apply gamma and beta affine transform
    
    mean, variance =compute_layer_norm_mean_and_variance(x)

    x_hat= (x- mean) /(torch.sqrt(variance+eps))

    output= gamma * x_hat + beta

    return output

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    # TODO: combine the residual with the sublayer output and layer-normalize the result.
    total_input= residual_input + sublayer_output

    answer= normalize_and_scale_with_gamma_beta(total_input,gamma, beta, eps=1e-5)
    
    return answer

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    # TODO: multiply x by the boolean keep_mask and rescale by 1/keep_prob.
    
    return (x * keep_mask)/ keep_prob

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # Self-attention: query, key, value all come from x
    attn_out = assemble_multi_head_attention_forward(
        x, x, x,
        w_q, w_k, w_v, w_o,
        num_heads,
        src_mask
    )

    # Residual add + layer norm
    out = apply_residual_add_and_norm(x, attn_out, gamma, beta)
    return out

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on x and wrap it with residual add-and-norm.
    output=position_wise_feed_forward_network(x, w1, b1, w2, b2)
    output=apply_residual_add_and_norm(output, x, gamma, beta, eps=1e-5)
    return output

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    # TODO: chain the self-attention sublayer and the feed-forward sublayer using layer_params.

    output=encoder_layer_self_attention_sublayer(x,layer_params['w_q'],layer_params['w_k'],layer_params['w_v'],layer_params['w_o'], layer_params['attn_gamma'],layer_params['attn_beta'], num_heads, src_mask)
    output=encoder_layer_feed_forward_sublayer(output,layer_params['w1'], layer_params['b1'], layer_params['w2'], layer_params['b2'], layer_params['ffn_gamma'], layer_params['ffn_beta'])

    return output

# Step 42 - stack_encoder_layers
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    # TODO: sequentially apply each encoder layer to the running hidden state and return the final tensor.

    hidden=x
    
    for params in encoder_layer_params_list:

        hidden = assemble_encoder_layer(hidden, params, num_heads, src_mask)
    
    return hidden

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    # TODO: run masked multi-head self-attention on y and wrap with residual add-and-norm.

    attn_out=assemble_multi_head_attention_forward(y,y, y, w_q, w_k, w_v, w_o, num_heads, tgt_mask)
    output=apply_residual_add_and_norm(y, attn_out, gamma, beta)

    return output

# Step 44 - decoder_layer_cross_attention_sublayer
import torch

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head cross-attention (Q from y, K/V from encoder_output) and wrap with add-and-norm
    if src_mask is not None:
        src_mask = src_mask[:, None, None, :] 
    
    attn_out=assemble_multi_head_attention_forward(y,encoder_output, encoder_output, w_q, w_k, w_v, w_o, num_heads, src_mask)
    output=apply_residual_add_and_norm(y, attn_out, gamma, beta)

    return output

# Step 45 - decoder_layer_feed_forward_sublayer
import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on y and wrap it with residual add-and-norm

    output=position_wise_feed_forward_network(y, w1, b1, w2, b2)
    output=apply_residual_add_and_norm(output, y, gamma, beta, eps=1e-5)
    
    return output

# Step 46 - assemble_decoder_layer
def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    def pick_param(*exact_names, required_parts=None, exclude_parts=None):
        for name in exact_names:
            if name in layer_params:
                return layer_params[name]

        if required_parts is not None:
            for k, v in layer_params.items():
                if all(part in k for part in required_parts):
                    if exclude_parts is None or all(part not in k for part in exclude_parts):
                        return v

        raise KeyError(exact_names[0] if exact_names else str(required_parts))

    # 1) masked self-attention
    output = decoder_layer_masked_self_attention_sublayer(
        y,
        pick_param(
            "masked_self_attention_w_q", "masked_self_w_q", "self_attn_w_q", "w_q",
            required_parts=("self", "q"),
            exclude_parts=("cross",)
        ),
        pick_param(
            "masked_self_attention_w_k", "masked_self_w_k", "self_attn_w_k", "w_k",
            required_parts=("self", "k"),
            exclude_parts=("cross",)
        ),
        pick_param(
            "masked_self_attention_w_v", "masked_self_w_v", "self_attn_w_v", "w_v",
            required_parts=("self", "v"),
            exclude_parts=("cross",)
        ),
        pick_param(
            "masked_self_attention_w_o", "masked_self_w_o", "self_attn_w_o", "w_o",
            required_parts=("self", "o"),
            exclude_parts=("cross",)
        ),
        pick_param(
            "masked_self_attention_gamma", "masked_self_gamma", "self_attn_gamma", "attn_gamma",
            required_parts=("self", "gamma"),
            exclude_parts=("cross",)
        ),
        pick_param(
            "masked_self_attention_beta", "masked_self_beta", "self_attn_beta", "attn_beta",
            required_parts=("self", "beta"),
            exclude_parts=("cross",)
        ),
        num_heads,
        tgt_mask,
    )

    # 2) cross-attention
    output = decoder_layer_cross_attention_sublayer(
        output,
        encoder_output,
        pick_param(
            "cross_attention_w_q", "cross_w_q", "cross_attn_w_q",
            required_parts=("cross", "q")
        ),
        pick_param(
            "cross_attention_w_k", "cross_w_k", "cross_attn_w_k",
            required_parts=("cross", "k")
        ),
        pick_param(
            "cross_attention_w_v", "cross_w_v", "cross_attn_w_v",
            required_parts=("cross", "v")
        ),
        pick_param(
            "cross_attention_w_o", "cross_w_o", "cross_attn_w_o",
            required_parts=("cross", "o")
        ),
        pick_param(
            "cross_attention_gamma", "cross_gamma", "cross_attn_gamma", "attn_gamma",
            required_parts=("cross", "gamma")
        ),
        pick_param(
            "cross_attention_beta", "cross_beta", "cross_attn_beta", "attn_beta",
            required_parts=("cross", "beta")
        ),
        num_heads,
        src_mask,
    )

    # 3) feed-forward network
    output = decoder_layer_feed_forward_sublayer(
        output,
        pick_param("w1", required_parts=("w1",)),
        pick_param("b1", required_parts=("b1",)),
        pick_param("w2", required_parts=("w2",)),
        pick_param("b2", required_parts=("b2",)),
        pick_param("ffn_gamma", required_parts=("ffn", "gamma")),
        pick_param("ffn_beta", required_parts=("ffn", "beta")),
    )

    return output

# Step 47 - stack_decoder_layers
def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    # TODO: sequentially apply each decoder layer to the running target hidden state.
 

    hidden=y

    for layer_params in decoder_layer_params_list:
        hidden=assemble_decoder_layer(hidden, encoder_output, layer_params, num_heads, src_mask, tgt_mask)
    
    return hidden

# Step 48 - apply_final_output_projection
def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    # TODO: project decoder hidden states (B, T, D) to vocabulary logits (B, T, V).

    output= decoder_output @ output_projection_weight.T 

    if output_projection_bias is not None:
        output += output_projection_bias
    
    return output

# Step 49 - tie_output_projection_to_token_embeddings
import torch

def tie_output_projection_to_token_embeddings(token_embedding_weight):
    """Return an output projection weight that shares storage with token_embedding_weight.

    Input shape: (vocab_size, d_model). Output shape: (d_model, vocab_size).
    """
    # TODO: return an output projection weight tied to the token embedding matrix
    
    return token_embedding_weight.T

# Step 50 - apply_log_softmax_over_vocab
import torch

import torch.nn.functional as F
def apply_log_softmax_over_vocab(logits):
    # TODO: Convert decoder logits (B, T, V) into log probabilities over the vocabulary axis.
    
    return F.log_softmax(logits, dim=-1)

# Step 51 - run_transformer_forward
def run_transformer_forward(src_token_ids, tgt_token_ids, model_params, num_heads, pad_id):
    src_embedding = model_params.get("src_embedding", model_params.get("token_embedding"))
    tgt_embedding = model_params.get("tgt_embedding", model_params.get("token_embedding"))
    encoder_layers = model_params["encoder_layers"]
    decoder_layers = model_params["decoder_layers"]
    output_projection = model_params["output_projection"]

    d_model = src_embedding.shape[1]

    # 1) Embed source and target token ids
    src_embeddings = src_embedding[src_token_ids]
    tgt_embeddings = tgt_embedding[tgt_token_ids]

    # 2) Scale embeddings
    src_embeddings = scale_embeddings_by_sqrt_d_model(src_embeddings, d_model)
    tgt_embeddings = scale_embeddings_by_sqrt_d_model(tgt_embeddings, d_model)

    # 3) Add positional encodings
    src_len = src_token_ids.shape[1]
    tgt_len = tgt_token_ids.shape[1]

    src_pos_enc = build_sinusoidal_positional_encoding(src_len, d_model)
    tgt_pos_enc = build_sinusoidal_positional_encoding(tgt_len, d_model)

    src_embeddings = add_positional_encoding_to_embeddings(src_embeddings, src_pos_enc)
    tgt_embeddings = add_positional_encoding_to_embeddings(tgt_embeddings, tgt_pos_enc)

    # 4) Build masks
    src_mask = build_padding_mask(src_token_ids, pad_id)
    tgt_padding_mask = build_padding_mask(tgt_token_ids, pad_id)
    tgt_causal_mask = build_causal_mask(tgt_len)
    tgt_mask = combine_padding_and_causal_masks(tgt_padding_mask, tgt_causal_mask)

    # 5) Run encoder stack
    encoder_output = stack_encoder_layers(src_embeddings, encoder_layers, num_heads, src_mask)

    # 6) Run decoder stack
    decoder_output = stack_decoder_layers(
        tgt_embeddings,
        encoder_output,
        decoder_layers,
        num_heads,
        src_mask,
        tgt_mask
    )

    # 7) Project to vocabulary logits
    logits = apply_final_output_projection(decoder_output, output_projection, None)

    # 8) Convert logits to log probabilities
    log_probs = apply_log_softmax_over_vocab(logits)

    return log_probs

# Step 52 - init_encoder_layer_parameters
import torch
import math

def init_encoder_layer_parameters(d_model, num_heads, d_ff):

    """Return a dict of leaf tensors with requires_grad=True for one encoder layer."""
    # TODO: allocate w_q, w_k, w_v, w_o, w1, b1, w2, b2, attn_gamma, attn_beta, ffn_gamma, ffn_beta.


    params={
        'w_q': torch.randn(d_model, d_model, dtype=torch.float32, requires_grad=True),
        'w_k': torch.randn(d_model, d_model, dtype=torch.float32, requires_grad=True),
        'w_v':torch.randn(d_model, d_model, dtype=torch.float32, requires_grad=True),
        'w_o':torch.randn(d_model, d_model, dtype=torch.float32, requires_grad=True),

        'w1': torch.rand(d_model,d_ff,dtype=torch.float32, requires_grad=True),
        'b1': torch.zeros(d_ff,dtype=torch.float32, requires_grad=True),   
        'w2': torch.rand(d_ff,d_model,dtype=torch.float32, requires_grad=True),
        'b2': torch.zeros(d_model,dtype=torch.float32, requires_grad=True),

        'attn_gamma': torch.ones(d_model,dtype=torch.float32, requires_grad=True),
        'attn_beta': torch.zeros(d_model,dtype=torch.float32, requires_grad=True),

        'ffn_gamma':torch.ones(d_model,dtype=torch.float32, requires_grad=True),
        'ffn_beta':torch.zeros(d_model,dtype=torch.float32, requires_grad=True)  

    }

    return params

# Step 53 - init_decoder_layer_parameters
import torch

def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    """Return a dict of requires_grad tensors for one decoder layer."""
    params = {
        # Masked self-attention
        "w_q_self": torch.randn(d_model, d_model, dtype=torch.float32, requires_grad=True),
        "w_k_self": torch.randn(d_model, d_model, dtype=torch.float32, requires_grad=True),
        "w_v_self": torch.randn(d_model, d_model, dtype=torch.float32, requires_grad=True),
        "w_o_self": torch.randn(d_model, d_model, dtype=torch.float32, requires_grad=True),

        # Cross-attention
        "w_q_cross": torch.randn(d_model, d_model, dtype=torch.float32, requires_grad=True),
        "w_k_cross": torch.randn(d_model, d_model, dtype=torch.float32, requires_grad=True),
        "w_v_cross": torch.randn(d_model, d_model, dtype=torch.float32, requires_grad=True),
        "w_o_cross": torch.randn(d_model, d_model, dtype=torch.float32, requires_grad=True),

        # Feed-forward network
        "w1": torch.randn(d_model, d_ff, dtype=torch.float32, requires_grad=True),
        "b1": torch.zeros(d_ff, dtype=torch.float32, requires_grad=True),
        "w2": torch.randn(d_ff, d_model, dtype=torch.float32, requires_grad=True),
        "b2": torch.zeros(d_model, dtype=torch.float32, requires_grad=True),

        # LayerNorm for masked self-attention
        "self_gamma": torch.ones(d_model, dtype=torch.float32, requires_grad=True),
        "self_beta": torch.zeros(d_model, dtype=torch.float32, requires_grad=True),

        # LayerNorm for cross-attention
        "cross_gamma": torch.ones(d_model, dtype=torch.float32, requires_grad=True),
        "cross_beta": torch.zeros(d_model, dtype=torch.float32, requires_grad=True),

        # LayerNorm for FFN
        "ffn_gamma": torch.ones(d_model, dtype=torch.float32, requires_grad=True),
        "ffn_beta": torch.zeros(d_model, dtype=torch.float32, requires_grad=True),
    }

    return params

# Step 54 - init_embedding_and_projection_parameters
import torch

def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):
    """Allocate src/tgt embeddings and output projection (optionally tied)."""
    # TODO: allocate three (vocab_size, d_model) tensors with requires_grad=True

    src_embedding=torch.rand(vocab_size,d_model,requires_grad=True)
    tgt_embedding=torch.rand(vocab_size,d_model,requires_grad=True)

    if tie_weights:
        output_projection= tgt_embedding
    else:
        output_projection=torch.rand(vocab_size,d_model,requires_grad=True)

    params={
        'src_embedding': src_embedding,
        'tgt_embedding': tgt_embedding,
        'output_projection':output_projection
    }

    return params

# Step 55 - collect_model_parameters_into_list
import torch

def collect_model_parameters_into_list(encoder_layer_params, decoder_layer_params, embedding_params):
    """Collect all unique trainable parameters into a flat list."""
    
    params = []
    seen = set()

    # Encoder layer parameters
    for layer in encoder_layer_params:
        for tensor in layer.values():
            if id(tensor) not in seen:
                params.append(tensor)
                seen.add(id(tensor))

    # Decoder layer parameters
    for layer in decoder_layer_params:
        for tensor in layer.values():
            if id(tensor) not in seen:
                params.append(tensor)
                seen.add(id(tensor))

    # Embedding / projection parameters
    for tensor in embedding_params.values():
        if id(tensor) not in seen:
            params.append(tensor)
            seen.add(id(tensor))

    return params

# Step 56 - shift_targets_right_with_start_token
import torch

def shift_targets_right_with_start_token(target_ids, start_token_id):
    # 1. Get the batch size from the target_ids tensor
    batch_size = target_ids.shape[0]
    
    # 2. Create a tensor of start tokens with shape [batch_size, 1]
    # We ensure it shares the same device and data type as target_ids
    start_tokens = torch.full(
        (batch_size, 1), 
        start_token_id, 
        dtype=target_ids.dtype, 
        device=target_ids.device
    )
    
    # 3. Concatenate the start tokens with all but the last column of target_ids
    return torch.cat((start_tokens, target_ids[:, :-1]), dim=1)

# Step 57 - compute_noam_learning_rate
def compute_noam_learning_rate(step, d_model, warmup_steps):
    # TODO: return the Noam warmup learning rate for the given step.

    return (d_model ** (-1/2)) * min( step ** (-1/2), step * (warmup_steps ** (-3/2)))

# Step 58 - build_uniform_smoothing_distribution
import torch

def build_uniform_smoothing_distribution(shape, vocab_size, epsilon):
    # TODO: return a float tensor of `shape` filled with epsilon / (vocab_size - 2).
    
    return torch.full(shape, epsilon / (vocab_size - 2))

# Step 59 - set_confidence_on_gold_tokens
import torch

def set_confidence_on_gold_tokens(smoothed_distribution, gold_token_ids, confidence):
    out = smoothed_distribution.clone()
    out.scatter_(-1, gold_token_ids.unsqueeze(-1), confidence)
    return out

# Step 60 - zero_pad_column_and_pad_token_rows
import torch

def zero_pad_column_and_pad_token_rows(smoothed_distribution, gold_token_ids, pad_id):
    out = smoothed_distribution.clone()

    # Zero the PAD column
    out[:, :, pad_id] = 0

    # Zero rows whose gold token is PAD
    pad_mask = (gold_token_ids == pad_id)
    out[pad_mask] = 0

    return out

# Step 61 - compute_label_smoothed_kl_loss
import torch

def compute_label_smoothed_kl_loss(log_probabilities, smoothed_distribution):
    """Return the summed KL loss over all (batch, time, vocab) entries."""
    # TODO: combine log_probabilities with the smoothed target distribution into a scalar loss
    loss = -(smoothed_distribution * log_probabilities).sum()

    return loss + 0.0

# Step 62 - average_loss_over_non_pad_tokens
import torch

def average_loss_over_non_pad_tokens(total_loss, gold_token_ids, pad_id):
    # TODO: divide total_loss by the count of non-pad tokens in gold_token_ids
    
    nod_pad= (gold_token_ids != pad_id).sum()
    return total_loss / max( nod_pad,1)

# Step 63 - compute_token_accuracy_ignoring_pad
import torch

def compute_token_accuracy_ignoring_pad(log_probabilities, gold_token_ids, pad_id):
    # TODO: argmax over vocab, compare to gold, average over non-pad positions only
    pred = log_probabilities.argmax(dim=-1)
    mask = gold_token_ids != pad_id
    correct = (pred == gold_token_ids)
    correct = correct[mask].sum()
    total = mask.sum()
    accuracy = correct / (total + 0.00002)

    return accuracy

# Step 64 - initialize_adam_optimizer_state
import torch

def initialize_adam_optimizer_state(parameter_list):
    """Allocate Adam m, v zero buffers and a step counter t=0."""
    # TODO: allocate zero buffers for first and second moments, plus step counter

    state={
        't':0,
        'm':[],
        'v':[]
    }

    for tensor in parameter_list:

        state['m'].append(torch.zeros_like(tensor))
        state['v'].append(torch.zeros_like(tensor))
    
    return state

# Step 65 - update_adam_first_moment
import torch

def update_adam_first_moment(m_prev, grad, beta1):
    """Return m_t = beta1 * m_prev + (1 - beta1) * grad."""
    # TODO: apply the Adam first-moment EMA update and return the new tensor

    return beta1 * m_prev + (1-beta1)* grad

# Step 66 - update_adam_second_moment
import torch

def update_adam_second_moment(v_prev, grad, beta2):
    """Return v_t = beta2 * v_prev + (1 - beta2) * grad ** 2."""
    # TODO: apply Adam's EMA update for the second moment of the gradient
    
    return beta2 * v_prev + (1- beta2) * grad ** 2

# Step 67 - apply_adam_bias_correction
import torch

def apply_adam_bias_correction(m_t, v_t, beta1, beta2, step):
    """Return bias-corrected (m_hat, v_hat) for Adam at the given step."""
    # TODO: divide each moment by (1 - beta**step) using its respective beta
    
    m_hat= (m_t)/ (1- beta1 ** step)
    v_hat= (v_t) / (1- beta2 ** step)

    return (m_hat,v_hat)

# Step 69 - apply_adam_step_to_all_parameters
import torch

def apply_adam_step_to_all_parameters(
    parameter_list,
    optimizer_state,
    learning_rate,
    beta1=0.9,
    beta2=0.98,
    epsilon=1e-9,
):
    # Increment optimizer step
    optimizer_state["t"] += 1
    t = optimizer_state["t"]

    # Loop through every parameter
    for i, param in enumerate(parameter_list):

        # Skip parameters with no gradient
        if param.grad is None:
            continue

        grad = param.grad

        # Update first moment
        optimizer_state["m"][i] = (
            beta1 * optimizer_state["m"][i]
            + (1 - beta1) * grad
        )

        # Update second moment
        optimizer_state["v"][i] = (
            beta2 * optimizer_state["v"][i]
            + (1 - beta2) * (grad ** 2)
        )

        # Bias correction
        m_hat = optimizer_state["m"][i] / (1 - beta1 ** t)
        v_hat = optimizer_state["v"][i] / (1 - beta2 ** t)

        # Compute Adam update
        delta = learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)

        # Update parameter in-place
        param.data -= delta

    return optimizer_state

# Step 70 - zero_all_parameter_gradients
import torch

def zero_all_parameter_gradients(parameter_list):
    """Clear the .grad of every parameter tensor before the next backward pass."""
    # TODO: clear the accumulated gradient on every parameter tensor in the list

    for params in parameter_list:

        params.grad= None

# Step 71 - compute_batch_training_loss
def compute_batch_training_loss(src_batch, tgt_batch, model_params, config):
    # Read config
    pad_id = config["pad_id"]
    start_id = config["start_id"]
    num_heads = config["num_heads"]
    smoothing = config["smoothing"]
    vocab_size = config["vocab_size"]

    # Teacher-forced decoder input
    decoder_input = shift_targets_right_with_start_token(tgt_batch, start_id)

    # Forward pass: returns log probabilities (B, T, V)
    log_probs = run_transformer_forward(
        src_batch,
        decoder_input,
        model_params,
        num_heads=num_heads,
        pad_id=pad_id,
    )

    # Build label-smoothed target distribution
    smoothed = build_uniform_smoothing_distribution(
        log_probs.shape,
        vocab_size=vocab_size,
        epsilon=smoothing,
    )
    smoothed = set_confidence_on_gold_tokens(
        smoothed,
        tgt_batch,
        confidence=1.0 - smoothing,
    )
    smoothed = zero_pad_column_and_pad_token_rows(smoothed, tgt_batch, pad_id)

    # KL-style loss over all positions, then average over non-pad tokens
    total_loss = compute_label_smoothed_kl_loss(log_probs, smoothed)
    loss = average_loss_over_non_pad_tokens(total_loss, tgt_batch, pad_id)

    return loss

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

