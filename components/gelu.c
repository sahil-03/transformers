#include "gelu.h" 
#include <cmath> 
#include <math.h>

GELU::GELU(bool approx): approx(approx) {}
GELU::~GELU() {}


inline float phi_x(float x) {
    return 0.5 * (1.0f + erf(x / sqrtf(2.0f)));
}

inline float phi_x_approx(float x) { 
    return 0.5 * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715 * x * x * x)));
}

inline float D_phi_x(float x) {
    return exp(-0.5 * x * x) / sqrtf(2.0f * M_PI);
}

inline float D_phi_x_approx(float x) { 
    float sech = sechf(sqrtf(2 / M_PI) * (x + 0.044715 * x * x * x)); 
    return 0.5 * sech * sech * sqrtf(2 / M_PI) * (1 + 3 * 0.044715 * x * x);
}


void GELU::gelu_forward_approx(float* X, float* Y, int N) {
    for (int i = 0; i < N; i++) {
        float x = X[i];
        Y[i] = x * phi_x_approx(x);
    }
}


void GELU::gelu_forward(float* X, float* Y, int N) {
    if (this->approx) {
        gelu_forward_approx(X, Y, N);    
    } else {
        for (int i = 0; i < N; i++) {
            float x = X[i];
            Y[i] = x * phi_x(x);
        }
    }
}


void GELU::gelu_backward_approx(float* X, float* DX, float* DY, int N) {
    for (int i = 0; i < N; i++) {
        float x = X[i];
        float Dgelu_x = phi_x_approx(x) + x * D_phi_x_approx(x);
        DX[i] = Dgelu_x *  DY[i];
    }
}


void GELU::gelu_backward(float* X, float* DX, float* DY, int N) {
    if (this->approx) {
        gelu_backward_approx(X, DX, DY, N);
    } else {
        for (int i = 0; i < N; i++) {
            float x = X[i];
            float Dgelu_x = phi_x(x) + x * D_phi_x(x); 
            DX[i] = Dgelu_x *  DY[i]; 
        }
    }
}

