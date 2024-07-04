#ifndef _GELU_H_
#define _GELU_H_


class GELU {
    public: 
        bool approx; 

        GELU(bool approx);  
        ~GELU();

        // functions 
        void gelu_forward(float* X, int B, int T, int C);
        void gelu_forward_approx(float* X, int B, int T, int C);
};

#endif 