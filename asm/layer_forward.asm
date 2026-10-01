; ------------------------------------------------------------
; void layer_forward(const float* W, const float* x, const float* b,
;                    float* out, size_t rows, size_t cols)
;   out = relu(W * x + b)
;   W is row-major, rows x cols. x has cols elements.
;   b and out have `rows` elements.
;
; Args:  rdi = W   rsi = x   rdx = b   rcx = out   r8 = rows   r9 = cols
; Ret:   nothing
; Calls: matvec, vec_add, relu_array
; Saved: rbx = out   r12 = b   r13 = rows   (callee-saved, pushed/popped)
; Stack: 3 pushes + return address = 32 bytes -> aligned for call
; ------------------------------------------------------------
global layer_forward
extern matvec
extern vec_add
extern relu_array

section .text
layer_forward:
    push rbx
    push r12
    push r13

    mov rbx, rcx ; rbx = out
    mov r12, rdx ; r12 = b
    mov r13, r8 ; r13 = rows 

    ; step 1: matvec(W, X, out, rows, cols)
    mov rdx, rbx ; arg 3 : out
    mov rcx, r13 ; arg 4 : rows
    mov r8, r9 ; arg 5 : cols
    call matvec 

    ; step 2: vec_add(out, b, out, rows)
    mov rdi, rbx ; arg 1 : out
    mov rsi, r12 ; arg 2 : b
    mov rdx, rbx ; arg 3 : out
    mov rcx, r13 ; arg 4 : rows
    call vec_add

    ; step 3: relu_array(out, rows) 
    mov rdi, rbx ; arg 1 : out
    mov rsi, r13 ; arg 2 : rows
    call relu_array

    pop r13
    pop r12
    pop rbx
    ret