#include <cstdio>
#include <cstddef>

extern "C" {
    float dot(const float* a, const float* b, size_t n);
    void  axpy(float alpha, const float* x, float* y, size_t n);
}

int main() {
    // training data: y = 2x
    float xs[] = {1, 2, 3, 4};
    float ys[] = {2, 4, 6, 8};

    float w[1]   = {0.0f};    // weight (starts knowing nothing)
    float b[1]   = {0.0f};    // bias
    float one[1] = {1.0f};    // used to update the bias with axpy
    float lr     = 0.01f;     // learning rate

    for (int epoch = 0; epoch < 200; epoch++) {
        float loss = 0.0f;

        for (int i = 0; i < 4; i++) {
            float x[1] = { xs[i] };

            float pred = dot(w, x, 1) + b[0];     // forward pass (asm)
            float err  = pred - ys[i];            // how wrong?
            loss += err * err;

            axpy(-lr * err, x,   w, 1);           // w -= lr * err * x   (asm)
            axpy(-lr * err, one, b, 1);           // b -= lr * err       (asm)
        }

        if (epoch % 20 == 0)
            std::printf("epoch %3d  loss %f  w %f  b %f\n", epoch, loss, w[0], b[0]);
    }

    float test[1] = {10.0f};
    std::printf("\nlearned w=%f b=%f\n", w[0], b[0]);
    std::printf("predict x=10 -> %f (want 20)\n", dot(w, test, 1) + b[0]);
    return 0;
}