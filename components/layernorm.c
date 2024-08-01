#include "../utils/tensor_operations.h"
#include <cmath> 



struct LayerNorm {
    LayerNorm() {}
    ~LayerNorm() {}


    void forward(float* X, float* Y, 
                        float* mean, float* rstd, 
                        float* gamma, float* beta, 
                        int B, int T, int C) {
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                float* x = threeDimRead(X, b, t, T, C);

                // Compute mean
                float mu = 0.0f; 
                for (int c = 0; c < C; c++) {
                    mu += x[c]; 
                }
                mu /= C;

                // Compute variance 
                float s = 0.0f; 
                for (int c = 0; c < C; c++) {
                    float mean_centered = x[c] - mu; 
                    s += mean_centered * mean_centered; 
                }
                s /= C; 
                
                // Cache mean and variance 
                float rs = 1.0f / sqrtf(s + EPSILON); 
                twoDimWrite(mean, b, t, T, mu); 
                twoDimWrite(rstd, b, t, T, rs);

                // Calculate layer norm
                float* y = threeDimRead(Y, b, t, T, C);
                for (int c = 0; c < C; c++) {
                    y[c] = ((x[c] - mu) * rs) * gamma[c] + beta[c]; 
                }
            }
        }
    }


    void backward(float* X, float* gamma, float* mean, float* rstd,
                         float* DX, float* DY, float* Dgamma, float* Dbeta, 
                         int B, int T, int C) {
    
        // Equation: Dx = 1/rstd * ((Dy * gamma) - (1/N x_hat (dot) (Dy * gamma)) * x_hat - 1/N Dy (dot) gamma)

        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) { 
                float* Dy = threeDimRead(DY, b, t, T, C);  
                float* Dx = threeDimRead(DX, b, t, T, C);
                float* x = threeDimRead(X, b, t, T, C);  
                float mu = twoDimRead(mean, b, t, T);
                float rs = twoDimRead(rstd, b, t, T);

                float norm_mean = 0.0f;  // (x - mu(x) / rs(x)) * 1/N
                float dy_gamma_mean = 0.0f;  // (Dy (dot) gamma) * 1/N
                for (int c = 0; c < C; c++) {
                    norm_mean += (x[c] - mu) * rs; 
                    dy_gamma_mean += Dy[c] * gamma[c];
                }
                norm_mean /= C; 
                dy_gamma_mean /= C; 

                for (int c = 0; c < C; c++) {
                    float dy_gamma_c = Dy[c] * gamma[c];  // Dy * gamma
                    float x_hat_c = (x[c] - mu) * rs;  // (x - mu(x) / rs(x))

                    Dgamma[c] += Dy[c] * x_hat_c;
                    Dbeta[c] += Dy[c]; 
                    Dx[c] += rs * (dy_gamma_c - (norm_mean * dy_gamma_c) * x_hat_c - dy_gamma_mean)
                }
            }
        }
    }
};