#include "../components/attention.h" 


void test_attention_forward() {

}

int main() {
    // Dims
    int B = 2; 
    int T = 3; 
    int C = 2; 
    int C3 = C * 3; 

    // Tesnors
    float* X = (float*)malloc(B * T * C3 * sizeof(float));
    float* Y = (float*)malloc(B * T * C * sizeof(float)); 
    float* attn = (float*)malloc(B * T * C * sizeof(float)); 

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < C3; c++) {
                int i = b * T * C3 + t * C3 + c;
                X[i] = b + t + c;
            }
        }
    }

    CausalSelfAttention a = new CausalSelfAttention(2, 2, false, false);
    a.causal_self_attention_forward(X, Y, attn, B, T, C);




}