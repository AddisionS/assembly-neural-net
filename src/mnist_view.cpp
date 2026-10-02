#include "mnist.h"

int main() {
    int n_img, n_lab;
    std::vector<float> imgs = load_images("data/train-images-idx3-ubyte", n_img);
    std::vector<int>   labs = load_labels("data/train-labels-idx1-ubyte", n_lab);
    std::printf("loaded %d images, %d labels\n", n_img, n_lab);

    const char* shades = " .:-=+*#%@";            // dark -> bright
    for (int k = 0; k < 3; k++) {                 // show the first 3 digits
        std::printf("\nlabel: %d\n", labs[k]);
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++)
                std::putchar(shades[(int)(imgs[k * 784 + r * 28 + c] * 9.0f)]);
            std::putchar('\n');
        }
    }
    return 0;
}