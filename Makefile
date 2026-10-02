ASM := $(wildcard asm/*.asm)
OBJ := $(ASM:asm/%.asm=build/%.o)
CXX_FLAGS := -O2

all: nn train train_mlp mnist_view

nn: src/main.cpp $(OBJ)
	g++ $(CXX_FLAGS) src/main.cpp $(OBJ) -o nn

train: src/train.cpp $(OBJ)
	g++ $(CXX_FLAGS) src/train.cpp $(OBJ) -o train

train_mlp: src/train_mlp.cpp src/mnist.h $(OBJ)
	g++ $(CXX_FLAGS) src/train_mlp.cpp $(OBJ) -o train_mlp

mnist_view: src/mnist_view.cpp src/mnist.h
	g++ $(CXX_FLAGS) src/mnist_view.cpp -o mnist_view

build/%.o: asm/%.asm
	@mkdir -p build
	nasm -felf64 $< -o $@

clean:
	rm -rf build nn train train_mlp mnist_view