//Test script to check the assembly code 
#include <cstdio>
#include <cstddef>
#include <cmath>

extern "C" {
    float add_f(float a, float b);
    float mul_f(float a, float b);
    float sum_array(const float* p, size_t n);
    float dot(const float* a, const float* b, size_t n);
    void  vec_add(const float* a, const float* b, float* out, size_t n);
    void  axpy(float alpha, const float* x, float* y, size_t n);
}

// Compare asm result with C++ reference; print PASS/FAIL
static void check(const char* name, float got, float want) {
    bool ok = std::fabs(got - want) < 1e-5f;
    std::printf("%-10s got %-10f want %-10f %s\n", name, got, want, ok ? "PASS" : "FAIL");
}

int main() {
    float a[] = {1, 2, 3};
    float b[] = {4, 5, 6};

    check("add_f",     add_f(1.5f, 2.25f), 1.5f + 2.25f);
    check("mul_f",     mul_f(1.5f, 4.0f),  1.5f * 4.0f);
    check("sum_array", sum_array(a, 3),    6.0f);
    check("sum_empty", sum_array(a, 0),    0.0f);
    check("dot",       dot(a, b, 3),       32.0f);

    float out[3];
    vec_add(a, b, out, 3);
    check("vec_add0", out[0], 5.0f);
    check("vec_add1", out[1], 7.0f);
    check("vec_add2", out[2], 9.0f);

    float w[]    = {0.5f, 0.5f, 0.5f};
    float grad[] = {1.0f, 2.0f, 3.0f};
    axpy(-0.1f, grad, w, 3);
    check("axpy0", w[0], 0.4f);
    check("axpy1", w[1], 0.3f);
    check("axpy2", w[2], 0.2f);

    return 0;
}