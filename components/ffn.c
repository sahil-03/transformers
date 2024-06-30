#include "ffn.h"
#include "../utils/tensor_operations.h"

Linear::Linear(bool optim): optim (optim) {}
Linear::~Linear() {}


void Linear::linear_forward(float* X, float* Y, 
                            float* W, float* bias, 
                            int B, int T, int C, int OC) {
    if (!this->optim) {
        naive_matmul_forward(X, Y, W, bias, B, T, C, OC);
    } else {
        // TODO: implement optimized matmul (forward)
    }
}

void Linear::linear_backward(float* X, float* W, 
                             float* DX, float* DY, float* DW, float* Dbias,
                             int B, int T, int C, int OC) {
    if (!this->optim) {
        naive_matmul_backward(X, W, DX, DY, DW, Dbias, B, T, C, OC);
    } else {
        // TODO: implement optimized matmul (backward)
    }

}