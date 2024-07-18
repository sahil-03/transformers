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
    int h_size = C / this->n_head; 

    for (int b = 0; b < B; b++) { 
        for (int t1 = 0; t1 < T; t1++) {
            for (int h = 0; h < this->n_head; h++) {
                float* q = QKV + b * T * C3 + t1 * C3 + h * h_size; 
                float* attn_bth = attn + b * this->n_head * T * T + h * T * T + t1 * T;

                // QKt / sqrt(head size)
                float max_val = -FLT_MAX;
                for (int t2 = 0; t2 < T; t2++) {
                    float* k = QKV + b * T * C3 + t2 * C3 + h * h_size + C; 
                    float val = 0.0f;
                    for (int i = 0;i  < h_size; i++) {
                        val += q[i] * k[i];
                    }
                    val *= (1.0f / sqrtf(h_size));
                    max_val = std::max(max_val, val);
                    attn_bth[t2] = val;
                }

                // softmax 
                float sum = 0.0f; 
                for (int t2 = 0; t2 < T; t2++) {
                    float exp = expf(attn_bth[t2] - max_val); 
                    sum += exp; 
                    attn_bth[t2] = exp;
                }
                for (int t2 = 0; t2 < T; t2++) {
                    attn_bth[t2] *= (1.0f / sum) * (t2 <= t1);  // apply the causal attention mask (compiler optimization)
                }

                // output
                float* y = Y + b * T * C + t1 * C + h * h_size; 
                for (int i = 0; i < h_size; i++) y[i] = 0.0f;
                for (int t2 = 0; t2 < T; t2++) {
                    float* v = QKV + b * T * C3 + t2 * C3 + h * h_size + C2; 
                    float attn_val = attn_bth[t2]; 
                    for (int i = 0; i < h_size; i++) {
                        y[i] += attn_val * v[i];
                    }
                }
            }
        }
    }
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
    for (int i = 0; i < B * T * C3; i++) {
        X[i] = v++;
        std::cout << X[i] << std::endl;
    }

    std::cout << "performing attention" << std::endl;
    CausalSelfAttention a(2, 2, false, false);
    a.causal_self_attention_forward(X, Y, attn, B, T, C);

    std::cout << "printing output" << std::endl;
    for (int i = 0; i < B * T * C; i++) std::cout << Y[i] << std::endl;

    free(X);
    free(Y);
    free(attn);


}