#ifndef _ATTENTION_H_ 
#define _ATTENTION_H_

class CausalSelfAttention {
    public: 
        int n_embd;
        int n_head; 


        CausalSelfAttention(int n_embd, int n_head); 
        ~CausalSelfAttention();

        void causal_self_attention_forward(); 
        void causal_self_attention_backward(); 
};

#endif 