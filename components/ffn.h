#ifndef _FFN_H_ 
#define _FFN_H_

class Linear {
    public: 
        bool optim; 

        Linear(bool optim); 
        ~Linear(); 

        void linear_forward(float* X, float* Y, 
                            float* W, float* bias, 
                            int B, int T, int C, int OC); 
        void linear_backward(float* X, float* W, 
                             float* DX, float* DY, float* DW, float* Dbias,
                             int B, int T, int C, int OC); 

};

#endif 