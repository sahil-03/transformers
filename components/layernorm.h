#ifndef _LAYERNORM_H_ 
#define _LAYERNORM_H_

#include <cmath>

const float EPSILON = 1e-5;

class LayerNorm {
    public:
        void forward(float* input, float* output, 
                     float* mean, float* sigma, 
                     float* gamma, float* beta, 
                     int B, int T, int C); 
        void backward();
    
};

inline float twoDimRead(float* x, int b, int t, int T) {
    return x[b * T + t];
}   

inline void twoDimWrite(float* x, int b, int t, int T, float val) {
    x[b * T + t] = val;
}

inline float* threeDimRead(float* x, int b, int t, int T, int C) {
    return x + b * T * C + t * C;
}

void LayerNorm::forward(float* input, float* output, 
                        float* mean, float* sigma, 
                        float* gamma, float* beta, 
                        int B, int T, int C) {

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* z = threeDimRead(input, b, t, T, C);

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

            // Cache mean and variance (for backward pass)
            twoDimWrite(mean, b, t, T, mu);
            twoDimWrite(sigma, b, t, T, s);

            // Compute normalization 
            float* out = threeDimRead(output, b, t, T, C);
            for (int c = 0; c < C; c++) {
                out[c] = (s * (z[c] - mu)) * gamma[c] + beta[c];
            }
        }
    }
}

void LayerNorm::backward() {

   // TODO

}


// Note --> try to understand what the input/output look like???

#endif 