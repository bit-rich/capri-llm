// Server-side kernels. Weights are int8 in both cases, so memory traffic is identical.
#include <stdint.h>
#include <stddef.h>

// Plaintext baseline: y = W x, int8 weights, int8 activations, int32 accumulate.
void gemv_i8_i8(const int8_t* W, const int8_t* x, int32_t* y, size_t dout, size_t din) {
    for (size_t i = 0; i < dout; i++) {
        const int8_t* w = W + i * din;
        int32_t acc = 0;
        for (size_t j = 0; j < din; j++) acc += (int32_t)w[j] * (int32_t)x[j];
        y[i] = acc;
    }
}

// Encrypted: y = W c mod 2^32, int8 weights, uint32 LWE ciphertext. Wraparound is the modulus.
void gemv_i8_u32(const int8_t* W, const uint32_t* c, uint32_t* y, size_t dout, size_t din) {
    for (size_t i = 0; i < dout; i++) {
        const int8_t* w = W + i * din;
        uint32_t acc = 0;
        for (size_t j = 0; j < din; j++) acc += (uint32_t)(int32_t)w[j] * c[j];
        y[i] = acc;
    }
}

// Batched versions (prefill / many tokens): Y[b] = W X[b]. X is (B, din) row-major.
void gemm_i8_i8(const int8_t* W, const int8_t* X, int32_t* Y, size_t dout, size_t din, size_t B) {
    for (size_t i = 0; i < dout; i++) {
        const int8_t* w = W + i * din;
        for (size_t b = 0; b < B; b++) {
            const int8_t* x = X + b * din;
            int32_t acc = 0;
            for (size_t j = 0; j < din; j++) acc += (int32_t)w[j] * (int32_t)x[j];
            Y[b * dout + i] = acc;
        }
    }
}
void gemm_i8_u32(const int8_t* W, const uint32_t* X, uint32_t* Y, size_t dout, size_t din, size_t B) {
    for (size_t i = 0; i < dout; i++) {
        const int8_t* w = W + i * din;
        for (size_t b = 0; b < B; b++) {
            const uint32_t* x = X + b * din;
            uint32_t acc = 0;
            for (size_t j = 0; j < din; j++) acc += (uint32_t)(int32_t)w[j] * x[j];
            Y[b * dout + i] = acc;
        }
    }
}
