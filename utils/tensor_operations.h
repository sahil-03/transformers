#ifndef _TENSOR_OPERATIONS_H 
#define _TENSOR_OPERATIONS_H


#include <cmath> 
#include <float.h> 

inline float twoDimRead(float* x, int b, int t, int T) {
    return x[b * T + t];
}

inline void twoDimWrite(float* x, int b, int t, int T, float val) {
    x[b * T + t] = val;
}

inline float* threeDimRead(float* x, int b, int t, int T, int C) {
    return x + b * T * C + t * C;
}

inline void threeDimWrite(float* x, int b, int t, int T, int C, float* val) {
    for (int c = 0; c < C; c++) {
        x[b * T * C + t * C + c] = val[c];
    }
}

inline void naive_matmul_forward(float* X, float* Y, 
                                 float* W, float* bias, 
                                 int B, int T, int C, int OC) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int bt = b * T + t; 
            for (int o = 0; o < OC; o++) {
                int val = (bias != NULL) * bias[o];
                for (int c = 0; c < C; c++) {
                    val += twoDimRead(X, bt, c, C) * twoDimRead(W, o, c, C);
                }
                twoDimWrite(Y, bt, o, OC, val);
            }
        }
    }
}

inline void naive_matmul_backward(float* X, float* W, 
                                  float* DX, float* DY, float* DW, float* Dbias, 
                                  int B, int T, int C, int OC) {
    // for (int b = 0; b < B; b++) {
    //     for (int t = 0; t < T; t++) {
    //         float* Dy = threeDimRead(DY, b, t, T, OC);
    //         float* Dx = threeDimRead(DX, b, t, T, C);
    //         float* x = threeDimRead(X, b, t, T, C);
    //         for (int o = 0; o < OC; o++) {
    //             float* Dw = twoDimRead(DW, o, 0, C);  
    //             float* w = twoDimRead(W, o, 0, C);
    //             float dy_o = Dy[o]; 
    //             Dbias[o] += (Dbias != NULL) * dy_o;
    //             for (int c = 0; c < C; c++) {
    //                 Dx[c] += w[c] * dy_o;
    //                 Dw[c] += x[c] * dy_o; 
    //             }
    //         }
    //     }
    // }
}


// inline void softmax(float* X, int N) {
//     double max = -10000; 
//     for (int i = 0; i < N; i++) { 
//         if (X[i] > max) max = X[i];
//     }

//     double sum = 0.0f; 
//     for (int i = 0; i < N; i++) {
//         X[i] = expf(X[i] - max);
//         sum += X[i];
//     }

//     for (int i = 0; i < N; i++) {
//         X[i] *= (1.0f / sum); 
//     }
// }

#endif 