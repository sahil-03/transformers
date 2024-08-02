#include "components/attention.c"
#include "components/ffn.c"
#include "components/gelu.c"
#include "components/layernorm.c"


typedef struct {
    uint32_t block_size; 
    uint32_t vocab_size; 
    uint32_t n_layer; 
    uint32_t n_head; 
    uint32_t n_embd; 
} GPTConfig;


typedef struct { 
    Linear c_fc = {}; 
    GELU gelu = {}; 
    Linear c_proj = {};
} MLP;


typedef struct {
    LayerNorm ln1 = {};
    CausalSelfAttention attn = {}; 
    LayerNorm ln2 = {}; 
    MLP mlp = {};
} Block;


struct GPT {
    GPTConfig config = {};

    // token embeddings 
    // positional embeddings 
    
    Block attn_heads[config.n_head];
    for (int i = 0; i < config.n_head; i++) {
        attn_heads[i] = {};
    }

    LayerNorm ln_f = {}; 
    Linear lm_head = {}; 
};