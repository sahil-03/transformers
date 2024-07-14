#include "attention.h" 
#include "../utils/tensor_operations.h"
#include <cmath> 
#include <iostream>


CausalSelfAttention::CausalSelfAttention(
                                            int n_embd, 
                                            int n_head, 
                                            bool use_kv_cache, 
                                            bool use_flash_attn
                                        ):  n_embd(n_embd), 
                                            n_head(n_head), 
                                            use_kv_cache(use_kv_cache), 
                                            use_flash_attn(use_flash_attn) {}


CausalSelfAttention::~CausalSelfAttention() {}


void CausalSelfAttention::causal_self_attention_forward(float* QKV, float* Y, float* attn, int B, int T, int C) {
   int C3 = C * 3;
   int C2 = C * 2; 

    // QK^T * 1/|K|
    std::cout << "qkT" << std::endl;
    for (int b = 0; b < B; b++) {
        for (int t1 = 0; t1 < T; t1++) {
            float* q = QKV + b * T * C3 + t1 * C3; 
            float* attn_bt1 = attn + b * T * T + t1 * T;
            for (int t2 = 0; t2 < T; t2++) {
                float* k = QKV + b * T * C3 + t2 * C3 + C; 
                float attn_val = 0.0f;
                for (int c = 0; c < C; c++) {
                    attn_val += q[c] * k[c];
                }
                attn_bt1[t2] = attn_val * (1 / sqrtf(C));
            }
        }
    }

    // Causal masking 
    // TODO!??

    std::cout << "softmax" << std::endl;
    softmax(attn, B * T * T); 
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < T; c++) {
                int i = b * T * T + t * T + c;
                std::cout << attn[i] << std::endl;
            }
        }
    }
    

    std::cout << "output" << std::endl;
    for (int b = 0; b < B; b++) {
        for (int t1 = 0; t1 < T; t1++) {
            float* attn_bt1 = attn + b * T * C + t1 * C;
            float* y = Y + b * T * C + t1 * C; 
            for (int t2 = 0; t2 < T; t2++) {
                float* v = QKV + b * T * C3 + t2 * C3 + C2; 
                float val = 0.0f;
                for (int c = 0; c < C; c++) {
                    val += attn_bt1[c] * v[c];
                }
                y[t2] = val;
            }
        }
    }
    std::cout << "done"<< std::endl;
}


int main() {
    // Dims
    int B = 1; 
    int T = 3; 
    int C = 2; 
    int C3 = C * 3; 

    // Tesnors
    float* X = (float*)malloc(B * T * C3 * sizeof(float));
    float* Y = (float*)malloc(B * T * C * sizeof(float)); 
    float* attn = (float*)malloc(B * T * T * sizeof(float)); 

    float v = 1.0f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < C3; c++) {
                int i = b * T * C3 + t * C3 + c;
                X[i] = v;
                v++;
                // std::cout << X[i] << std::endl;
            }
        }
    }

    std::cout << "performing attention" << std::endl;
    CausalSelfAttention a(2, 2, false, false);
    a.causal_self_attention_forward(X, Y, attn, B, T, C);

    std::cout << "printing output" << std::endl;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < C; c++) {
                int i = b * T * C + t * C + c;
                // std::cout << i << std::endl;
                std::cout << Y[i] << std::endl;
            }
        }
    }

    free(X);
    free(Y);
    free(attn);


}