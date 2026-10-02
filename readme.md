# Assembly Neural Network

A handwritten-digit classifier (MNIST) whose math is written in **x86-64 assembly (NASM)**. C++ is used only as a harness: loading files, shuffling, printing, and unit tests.

The goal was never to compete with PyTorch. It was to understand how neural-network operations turn into CPU instructions, starting from zero assembly and zero ML.

## Result

A 784 → 128 → 10 network (ReLU hidden layer, softmax output) trained with plain per-image SGD.

```
epoch  1  avg loss 0.2138  test accuracy 96.48%
epoch  2  avg loss 0.0981  test accuracy 97.36%
epoch  5  avg loss 0.0427  test accuracy 97.70%
epoch 10  avg loss 0.0134  test accuracy 97.85%
```

Accuracy is measured on the 10,000 MNIST test images, which are never used for training.

Every forward and backward step in the training loop is an assembly function. This includes `exp`, which is implemented from scratch (no libm).

## Project structure

```
asm/      20 assembly functions, one per file (NASM)
src/
  main.cpp         unit tests: every asm function checked against a known answer
  train.cpp        first experiment: one neuron learns y = 2x
  mnist.h          MNIST file loader
  mnist_view.cpp   prints digits as ASCII art to verify the data
  train_mlp.cpp    the digit classifier
data/     MNIST files (not committed)
Makefile
```

## Requirements

- Linux or WSL (x86-64)
- `nasm`, `g++`, `make`, `wget`

```
sudo apt install nasm g++ make wget
```

## Get the data

```
mkdir -p data && cd data
wget https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
cd ..
```

## Build and run

```
make              # assembles asm/*.asm into build/ and builds all programs
./nn              # unit tests, should end with ALL PASS
./mnist_view      # first 3 training digits as ASCII art (5, 0, 4)
./train           # one neuron learns y = 2x
./train_mlp       # trains the digit classifier (about 10 epochs)
```

Run everything from the project root, since data paths are relative.

## The assembly functions

All are scalar `float32` (SSE: `movss`, `addss`, `mulss`, ...), following the System V calling convention.

| Function | Does | Used in training |
|---|---|---|
| `add_f`, `mul_f` | scalar add / multiply | warm-up |
| `sum_array` | sum of an array | warm-up |
| `dot` | dot product | yes (via `matvec`) |
| `vec_add` | `out = a + b` | yes (bias) |
| `vec_sub` | `out = a - b` | yes (output error) |
| `vec_mul` | `out = a * b` | spare |
| `axpy` | `y += alpha * x` | yes (weight/bias update) |
| `relu` | `max(0, x)` | warm-up |
| `relu_array` | ReLU over an array | yes (via `layer_forward`) |
| `relu_grad` | 1 if x > 0 else 0 | spare |
| `relu_backward` | zero the error where ReLU was off | yes |
| `neuron_forward` | `relu(dot(w, x) + b)` | warm-up |
| `matvec` | matrix times vector | yes |
| `matvec_t` | transpose matrix times vector | yes (backprop) |
| `layer_forward` | `relu(W x + b)` | yes |
| `outer_update` | `W[r] -= lr * delta[r] * x` for every row | yes |
| `exp_f` | `e^x` via range reduction, polynomial, exponent bits | yes (via `softmax`) |
| `softmax` | scores to probabilities | yes |
| `argmax` | index of the largest value | yes (accuracy) |

## How training works

For each image:

1. **Forward:** `h = relu(W1 x + b1)`, then `scores = W2 h + b2`, then `probs = softmax(scores)`
2. **Output error:** `delta2 = probs - target` (target is one-hot)
3. **Backprop:** `delta1 = W2ᵀ delta2`, then zero it where the hidden neuron was off
4. **Update:** `W -= lr * delta * input` per layer, biases updated with `axpy`

Learning rate is 0.01 with He-style random initialization. The weight update is the `axpy` function called with `alpha = -lr * error`.

## Assembly concepts used, in the order they were learned

1. Calling convention: integer/pointer args in `rdi, rsi, rdx, rcx, r8, r9`, float args in `xmm0..`, returns in `rax` / `xmm0`
2. Loops with a pointer, a counter, and `jnz`
3. Loading and storing memory (`movss`)
4. Float comparisons: `maxss`, `cmpltss` masks, `comiss`
5. Constants in `.rodata` with `[rel ...]`
6. Calling your own functions, with stack space and `push` / `pop` of callee-saved registers
7. Indexed addressing and `lea`
8. Calling libm, then replacing it with a hand-written `exp` using `cvtss2si`, `movd`, and Horner's method

## Development routine

For every function: write the C++ reference, write the assembly, compare in `main.cpp`, commit.

## Possible next steps

- SIMD (`mulps` / `addps`, `vfmadd231ps`) in `dot` and `axpy`, the two functions that do nearly all the work
- Save and load trained weights
- A demo program that shows a random test digit and the network's prediction
- Mini-batches, a larger hidden layer, or Fashion-MNIST