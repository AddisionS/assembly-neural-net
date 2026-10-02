// Unit tests: every asm function checked against a known answer
#include <cstdio>
#include <cstddef>
#include <cmath>

extern "C" {
    float  add_f(float a, float b);
    float  mul_f(float a, float b);
    float  sum_array(const float* p, size_t n);
    float  dot(const float* a, const float* b, size_t n);
    void   vec_add(const float* a, const float* b, float* out, size_t n);
    void   vec_sub(const float* a, const float* b, float* out, size_t n);
    void   vec_mul(const float* a, const float* b, float* out, size_t n);
    void   axpy(float alpha, const float* x, float* y, size_t n);
    float  relu(float x);
    void   relu_array(float* x, size_t n);
    float  relu_grad(float x);
    void   relu_backward(float* delta, const float* z, size_t n);
    float  neuron_forward(const float* w, const float* x, float b, size_t n);
    void   matvec(const float* W, const float* x, float* out, size_t rows, size_t cols);
    void   matvec_t(const float* W, const float* v, float* out, size_t rows, size_t cols);
    void   layer_forward(const float* W, const float* x, const float* b,
                         float* out, size_t rows, size_t cols);
    void   outer_update(float lr, float* W, const float* delta, const float* x,
                        size_t rows, size_t cols);
    size_t argmax(const float* x, size_t n);
    float  exp_f(float x);
    void   softmax(const float* in, float* out, size_t n);
}

static int fails = 0;

static void check(const char* name, float got, float want) {
    bool ok = std::fabs(got - want) < 1e-5f;
    if (!ok) fails++;
    std::printf("%-14s got %-12f want %-12f %s\n", name, got, want, ok ? "PASS" : "FAIL");
}

int main() {
    float a[] = {1, 2, 3};
    float b[] = {4, 5, 6};

    check("add_f",     add_f(1.5f, 2.25f), 3.75f);
    check("mul_f",     mul_f(1.5f, 4.0f),  6.0f);
    check("sum_array", sum_array(a, 3),    6.0f);
    check("sum_empty", sum_array(a, 0),    0.0f);
    check("dot",       dot(a, b, 3),       32.0f);

    float out[3];
    vec_add(a, b, out, 3);
    check("vec_add0", out[0], 5.0f);
    check("vec_add1", out[1], 7.0f);
    check("vec_add2", out[2], 9.0f);

    float sa[] = {10, 20, 30};
    float sb[] = {1, 5, 40};
    float so[3];
    vec_sub(sa, sb, so, 3);
    check("vec_sub0", so[0], 9.0f);
    check("vec_sub1", so[1], 15.0f);
    check("vec_sub2", so[2], -10.0f);
    vec_mul(sa, sb, so, 3);
    check("vec_mul0", so[0], 10.0f);
    check("vec_mul1", so[1], 100.0f);
    check("vec_mul2", so[2], 1200.0f);

    float w[]    = {0.5f, 0.5f, 0.5f};
    float grad[] = {1.0f, 2.0f, 3.0f};
    axpy(-0.1f, grad, w, 3);
    check("axpy0", w[0], 0.4f);
    check("axpy1", w[1], 0.3f);
    check("axpy2", w[2], 0.2f);

    check("relu_pos",  relu(3.5f),  3.5f);
    check("relu_neg",  relu(-2.0f), 0.0f);
    check("relu_zero", relu(0.0f),  0.0f);

    float r[] = {-1.0f, 2.0f, -3.5f, 0.0f, 4.0f};
    relu_array(r, 5);
    check("relu_arr0", r[0], 0.0f);
    check("relu_arr1", r[1], 2.0f);
    check("relu_arr2", r[2], 0.0f);
    check("relu_arr3", r[3], 0.0f);
    check("relu_arr4", r[4], 4.0f);

    check("grad_pos",  relu_grad(3.5f),  1.0f);
    check("grad_neg",  relu_grad(-2.0f), 0.0f);
    check("grad_zero", relu_grad(0.0f),  0.0f);

    float rbd[] = {5, 6, 7, 8};
    float rbz[] = {1, -1, 0, 2};
    relu_backward(rbd, rbz, 4);
    check("relu_bw0", rbd[0], 5.0f);
    check("relu_bw1", rbd[1], 0.0f);
    check("relu_bw2", rbd[2], 0.0f);
    check("relu_bw3", rbd[3], 8.0f);

    float w3[] = {1, 2, 3};
    float x3[] = {4, 5, 6};
    check("neuron_pos",   neuron_forward(w3, x3, -1.0f, 3),   31.0f);
    check("neuron_neg",   neuron_forward(w3, x3, -100.0f, 3), 0.0f);
    float w2[] = {1, -1};
    float x2[] = {1, 2};
    check("neuron_clamp", neuron_forward(w2, x2, 0.5f, 2), 0.0f);

    float W[]  = {1, 2, 3,
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

    float mtv[] = {1, 10};
    float mto[3] = {99, 99, 99};          // garbage on purpose: must be zeroed
    matvec_t(W, mtv, mto, 2, 3);
    check("matvec_t0", mto[0], 41.0f);
    check("matvec_t1", mto[1], 52.0f);
    check("matvec_t2", mto[2], 63.0f);

    float lx[]  = {1, 1, 1};
    float lb[]  = {-10.0f, 1.0f};
    float lo[2];
    layer_forward(W, lx, lb, lo, 2, 3);
    check("layer0", lo[0], 0.0f);
    check("layer1", lo[1], 16.0f);
    float lb2[] = {0.5f, -20.0f};
    layer_forward(W, lx, lb2, lo, 2, 3);
    check("layer2", lo[0], 6.5f);
    check("layer3", lo[1], 0.0f);
    float lb3[] = {1.0f, 1.0f};
    layer_forward(W, lx, lb3, lo, 2, 3);
    check("layer4", lo[0], 7.0f);
    check("layer5", lo[1], 16.0f);

    float ouW[] = {1, 1, 1,
                   1, 1, 1};
    float oud[] = {1.0f, 2.0f};
    float oux[] = {1.0f, 2.0f, 3.0f};
    outer_update(0.5f, ouW, oud, oux, 2, 3);
    check("outer00", ouW[0],  0.5f);
    check("outer01", ouW[1],  0.0f);
    check("outer02", ouW[2], -0.5f);
    check("outer10", ouW[3],  0.0f);
    check("outer11", ouW[4], -1.0f);
    check("outer12", ouW[5], -2.0f);

    float am1[] = {0.1f, 0.7f, 0.2f};
    float am2[] = {-5.0f, -2.0f, -9.0f};
    float am3[] = {1.0f, 2.0f, 3.0f, 9.0f};
    float am4[] = {4.0f, 4.0f, 1.0f};
    float am5[] = {7.0f};
    check("argmax_mid",  (float)argmax(am1, 3), 1.0f);
    check("argmax_neg",  (float)argmax(am2, 3), 1.0f);
    check("argmax_last", (float)argmax(am3, 4), 3.0f);
    check("argmax_tie",  (float)argmax(am4, 3), 0.0f);
    check("argmax_one",  (float)argmax(am5, 1), 0.0f);

    check("exp_0",   exp_f(0.0f),  1.0f);
    check("exp_1",   exp_f(1.0f),  2.7182817f);
    check("exp_neg", exp_f(-1.0f), 0.36787945f);
    check("exp_2",   exp_f(2.0f),  7.389056f);
    check("exp_m5",  exp_f(-5.0f), 0.006737947f);

    float sm_in[] = {1.0f, 2.0f, 3.0f};
    float sm_out[3];
    softmax(sm_in, sm_out, 3);
    check("softmax0", sm_out[0], 0.09003057f);
    check("softmax1", sm_out[1], 0.24472847f);
    check("softmax2", sm_out[2], 0.66524096f);
    float sm_big[] = {1000.0f, 1000.0f};
    float sm_big_out[2];
    softmax(sm_big, sm_big_out, 2);
    check("softmax_big0", sm_big_out[0], 0.5f);
    check("softmax_big1", sm_big_out[1], 0.5f);

    std::printf("\n%s (%d failed)\n", fails == 0 ? "ALL PASS" : "SOME FAILED", fails);
    return fails != 0;
}