#ifndef _GELU_H_
#define _GELU_H_


class GELU {
    public: 
        bool approx; 

        GELU(bool approx);  
        ~GELU();

        void gelu_forward(float* X, float* Y, int N);
        void gelu_forward_approx(float* X, float* Y, int N);
        void gelu_backward(float* X, float* DX, float* DY, int N);
        void gelu_backward_approx(float* X, float* DX, float* DY, int N);
};

#endif 