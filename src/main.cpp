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
    float relu(float x);
    float relu_grad(float x);
    void  relu_array(float* x, size_t n);
    float neuron_forward(const float* w, const float* x, float b, size_t n);
    void matvec(const float* W, const float* x, float* out, size_t rows, size_t cols);
    void layer_forward(const float* W, const float* x, const float* b, float* out, size_t rows, size_t cols);
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

    check("relu_pos",  relu(3.5f),  3.5f);
    check("relu_neg",  relu(-2.0f), 0.0f);
    check("relu_zero", relu(0.0f),  0.0f);

    check("grad_pos",  relu_grad(3.5f),  1.0f);
    check("grad_neg",  relu_grad(-2.0f), 0.0f);
    check("grad_zero", relu_grad(0.0f),  0.0f);

    float r[] = {-1.0f, 2.0f, -3.5f, 0.0f, 4.0f};
    relu_array(r, 5);
    check("relu_arr0", r[0], 0.0f);
    check("relu_arr1", r[1], 2.0f);
    check("relu_arr2", r[2], 0.0f);
    check("relu_arr3", r[3], 0.0f);
    check("relu_arr4", r[4], 4.0f);

    float w3[] = {1, 2, 3};
    float x3[] = {4, 5, 6};
    check("neuron_pos", neuron_forward(w3, x3, -1.0f, 3),   31.0f);   
    check("neuron_neg", neuron_forward(w3, x3, -100.0f, 3), 0.0f);    

    float w2[] = {1, -1};
    float x2[] = {1, 2};
    check("neuron_clamp", neuron_forward(w2, x2, 0.5f, 2), 0.0f);   

    float W[] = {1, 2, 3,
                4, 5, 6};          
    float xv[] = {1, 1, 1};
    float mv[2];
    matvec(W, xv, mv, 2, 3);
    check("matvec0", mv[0], 6.0f);     
    check("matvec1", mv[1], 15.0f);    

    float xv2[] = {1, 0, -1};
    matvec(W, xv2, mv, 2, 3);
    check("matvec2", mv[0], -2.0f);    
    check("matvec3", mv[1], -2.0f);

    float LW[] = {1, 2, 3,
                4, 5, 6};         
    float lx[] = {1, 1, 1};
    float lb[] = {-10.0f, 1.0f};
    float lo[2];
    layer_forward(LW, lx, lb, lo, 2, 3);
    check("layer0", lo[0], 0.0f);      
    check("layer1", lo[1], 16.0f);     

    float lb2[] = {0.5f, -20.0f};
    layer_forward(LW, lx, lb2, lo, 2, 3);
    check("layer2", lo[0], 6.5f);     
    check("layer3", lo[1], 0.0f);    

    return 0;
}