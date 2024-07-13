#include "attention.h" 
#include "../utils/tensor_operations.h"
#include <cmath> 


CausalSelfAttention::CausalSelfAttention(
                                            int n_embd, 
                                            int n_head, 
                                            bool use_kv_cache, 
                                            bool use_flash_attn
                                        ):  n_embd(n_emd), 
                                            n_head(n_head), 
                                            use_kv_cache(use_kv_cache), 
                                            use_flash_attn(use_flash_attn) {}


CausalSelfAttention::~CausalSelfAttention() {}


CausalSelfAttention::causal_self_attention_forward(float* QKV, float* Y, float* attn, int B, int T, int C ) {
   int C_3 = C * 3;
   int C_2 = C * 2; 

    // QK^T * 1/|K|
    for (int b = 0; b < B; b++) {
        for (int t1 = 0; t1 < T; t1++) {
            float* q = QKV + b * T * C3 + t1 * C3; 
            float* attn_bt = attn + b * T * C3 + t1 * C3;
            for (int t2 = 0; t2 < T; t2++) {
                float* k = QKV + b * T * C3 + t1 * C3 + C: 
                for (int c = 0; c < C; c++) {
                    attn_bt[c] += q[c] * k[c];
                }
                attn_bt[c] *= (1 / sqrtf(C)); // k.size = 768 = C??
            }
        }
    }

    // Causal masking 
    // TODO!

    // Apply softmax 
    softmax(attn, B * T * C); 

    // Compute output (Y = attn @ v)
    for (int b = 0; b < B; b++) {
        for (int t1 = 0; t1 < T; t1++) {
            float* y = Y + b * T * C + t1 * C; 
            float* attn_bt = attn + b * T * C + t1 * C; 
            for (int t2 = 0; t2 < T; t2++) {
                float* v = QKV + b * T * C + t2 * C + C2; 
                for (int c = 0; c < C; c++) {
                    y[c] += attn_bt[c] * v[c]; 
                }
            }
        }
    }
}