#ifndef _ATTENTION_H_ 
#define _ATTENTION_H_

class CausalSelfAttention {
    public: 
        bool use_kv_cache; 
        bool use_flash_attn; 
        int n_embd;
        int n_head; 

        CausalSelfAttention(int n_embd, int n_head, bool use_kv_cache, bool use_flash_attn); 
        ~CausalSelfAttention();

        void causal_self_attention_forward(float* QKV, float* Y, float* attn, int B, int T, int C); 
        // void causal_self_attention_backward(); 
};

#endif 