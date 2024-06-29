#ifndef _TENSOR_OPERATIONS_H 
#define _TENSOR_OPERATIONS_H


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

#endif 