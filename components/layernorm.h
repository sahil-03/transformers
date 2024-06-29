#ifndef _LAYERNORM_H_ 
#define _LAYERNORM_H_


const float EPSILON = 1e-5;

class LayerNorm {
    public:
        // Constructor and destructor
        LayerNorm();
        ~LayerNorm();

        // Forward and backward pass
        void forward(float* X, float* Y, 
                     float* mean, float* rstd, 
                     float* gamma, float* beta, 
                     int B, int T, int C); 
        void backward(float* X, float* gamma, float* mean, float* rstd,
                      float* DX, float* DY, float* Dgamma, float* Dbeta, 
                      int B, int T, int C);
};

#endif 