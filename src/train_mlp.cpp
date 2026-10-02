#include "mnist.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

extern "C" {
    void   matvec(const float* W, const float* x, float* out, size_t rows, size_t cols);
    void   matvec_t(const float* W, const float* v, float* out, size_t rows, size_t cols);
    void   layer_forward(const float* W, const float* x, const float* b,
                         float* out, size_t rows, size_t cols);
    void   vec_add(const float* a, const float* b, float* out, size_t n);
    void   vec_sub(const float* a, const float* b, float* out, size_t n);
    void   axpy(float alpha, const float* x, float* y, size_t n);
    void   outer_update(float lr, float* W, const float* delta, const float* x,
                        size_t rows, size_t cols);
    void   relu_backward(float* delta, const float* z, size_t n);
    void   softmax(const float* in, float* out, size_t n);
    size_t argmax(const float* x, size_t n);
}

int main() {
    int n_train, n_test, n_tmp;
    std::vector<float> train_x = load_images("data/train-images-idx3-ubyte", n_train);
    std::vector<int>   train_y = load_labels("data/train-labels-idx1-ubyte", n_tmp);
    std::vector<float> test_x  = load_images("data/t10k-images-idx3-ubyte",  n_test);
    std::vector<int>   test_y  = load_labels("data/t10k-labels-idx1-ubyte",  n_tmp);

    const int IN = 784, HID = 128, OUT = 10;
    std::vector<float> W1(HID * IN), b1(HID, 0.0f);
    std::vector<float> W2(OUT * HID), b2(OUT, 0.0f);

    // random start, scaled to each layer's size (He initialization)
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d1(-std::sqrt(6.0f / IN),  std::sqrt(6.0f / IN));
    std::uniform_real_distribution<float> d2(-std::sqrt(6.0f / HID), std::sqrt(6.0f / HID));
    for (float& v : W1) v = d1(rng);
    for (float& v : W2) v = d2(rng);

    const float lr = 0.01f;
    std::vector<int> order(n_train);
    std::iota(order.begin(), order.end(), 0);

    for (int epoch = 1; epoch <= 10; epoch++) {
        std::shuffle(order.begin(), order.end(), rng);
        double loss = 0.0;

        for (int idx : order) {
            const float* x = &train_x[(size_t)idx * IN];
            int label = train_y[idx];

            float h[HID], scores[OUT], probs[OUT];
            float target[OUT] = {0}, delta2[OUT], delta1[HID];
            target[label] = 1.0f;

            // forward
            layer_forward(W1.data(), x, b1.data(), h, HID, IN);   // h = relu(W1 x + b1)
            matvec(W2.data(), h, scores, OUT, HID);               // scores = W2 h
            vec_add(scores, b2.data(), scores, OUT);              //        + b2
            softmax(scores, probs, OUT);
            loss += -logf(probs[label] + 1e-12f);

            // backward
            vec_sub(probs, target, delta2, OUT);                  // output error
            matvec_t(W2.data(), delta2, delta1, OUT, HID);        // blame for hidden layer
            relu_backward(delta1, h, HID);                        // off neurons get no blame

            // update (W2 only after delta1 is computed)
            outer_update(lr, W2.data(), delta2, h, OUT, HID);
            axpy(-lr, delta2, b2.data(), OUT);
            outer_update(lr, W1.data(), delta1, x, HID, IN);
            axpy(-lr, delta1, b1.data(), HID);
        }

        int correct = 0;
        for (int i = 0; i < n_test; i++) {
            float h[HID], scores[OUT];
            layer_forward(W1.data(), &test_x[(size_t)i * IN], b1.data(), h, HID, IN);
            matvec(W2.data(), h, scores, OUT, HID);
            vec_add(scores, b2.data(), scores, OUT);
            if ((int)argmax(scores, OUT) == test_y[i]) correct++;
        }
        std::printf("epoch %2d  avg loss %.4f  test accuracy %.2f%%\n",
                    epoch, loss / n_train, 100.0 * correct / n_test);
        std::fflush(stdout);
    }
    return 0;
}