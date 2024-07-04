#include "gelu.h" 
#include <cmath> 
#include <math.h>

GELU::GELU(bool approx): approx(approx) {}
GELU::~GELU() {}


void GELU::gelu_forward_approx(float* X, float* Y, int N) {
    for (int i = 0; i < N; i++) {
        float x = X[i];
        Y[i] = 0.5 * x * (1 + tanhf(sqrtf(2 / M_PI) * (x + 0.044715 * powf(x, 3))));
    }
}


void GELU::gelu_forward(float* X, float* Y, int N) {
    if (this->approx) {
        gelu_forward_approx(X, Y, N);    
    } else {
        for (int i = 0; i < N; i++) {
            float x = X[i];
            Y[i] = 0.5 * x * (1 + erf(x / sqrtf(2)));
        }
    }
}

void GELU::gelu_backward() {
    
}

