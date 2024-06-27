#ifndef _LAYERNORM_H_ 
#define _LAYERNORM_H_

#include <cmath>

const float EPSILON = 1e-5;

class LayerNorm {
    public:
        void forward(float* input, float* output, 
                     float* mean, float* rstd, 
                     float* gamma, float* beta, 
                     int B, int T, int C); 
        void backward();
    
};


void LayerNorm::forward(float* input, float* output, 
                        float* mean, float* rstd, 
                        float* gamma, float* beta, 
                        int B, int T, int C) {

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* z = input + b * T * C + t * C;

            // Compute mean 
            float mu = 0.0f; 
            for (int c = 0; c < C; c++) {
                mu += z[c];
            }
            mu /= C;

            // Compute variance 
            float s = 0.0f; 
            for (int c = 0; c < C; c++) {
                float d = z[c] - mu;
                s += d * d;
            }
            s = 1.0f / sqrtf((s / C) + EPSILON);

            // Cache mean and variance 
            mean[b * T + t] = mu / C;
            rstd[b * T + t] = s; 

            // Compute normalization 
            float* out_bt = output + b * T * C + t * C;
            for (int c = 0; c < C; c++) {
                out_bt[c] = (s * (z[c] - mu)) * gamma[c] + beta[c];
            }
        }
    }
}

void LayerNorm::backward() {
    // TODO 
}


// Note --> try to understand what the input/output look like???

#endif 