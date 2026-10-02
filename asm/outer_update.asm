; ------------------------------------------------------------
; void outer_update(float lr, float* W, const float* delta,
;                   const float* x, size_t rows, size_t cols)
;   W[r][c] -= lr * delta[r] * x[c]
;   W is row-major, rows x cols
;
; Args:  xmm0 = lr   rdi = W   rsi = delta   rdx = x
;        rcx = rows   r8 = cols
; Ret:   nothing
; Calls: axpy
; Saved: r12 = current row of W   r13 = delta pointer   r14 = x
;        r15 = rows remaining     rbx = cols
; Stack: 5 pushes + 16 bytes holding -lr (keeps rsp 16-aligned)
; ------------------------------------------------------------
global outer_update
extern axpy

section .text
outer_update:
    push  rbx
    push  r12
    push  r13
    push  r14
    push  r15
    sub   rsp, 16 ; space for -lr (16 keeps alignment)

    xorps xmm1, xmm1 ; xmm1 = 0.0
    subss xmm1, xmm0 ; xmm1 = 0 - lr = -lr
    movss [rsp], xmm1 ; save -lr (axpy will trash xmm registers)

    mov   r12, rdi ; r12 = W (current row)
    mov   r13, rsi ; r13 = delta pointer
    mov   r14, rdx ; r14 = x
    mov   r15, rcx ; r15 = rows remaining
    mov   rbx, r8 ; rbx = cols

    test  r15, r15 ; rows == 0 ?
    jz    .done
.loop:
    movss xmm0, [r13] ; xmm0 = delta[r]
    mulss xmm0, [rsp] ; xmm0 = delta[r] * (-lr)  = alpha
    mov   rdi, r14 ; axpy arg: x
    mov   rsi, r12 ; axpy arg: y = current row of W
    mov   rdx, rbx ; axpy arg: n = cols
    call  axpy ; row += alpha * x
    lea   r12, [r12 + rbx*4] ; next row of W
    add   r13, 4 ; next delta
    dec   r15 ; rows--
    jnz   .loop
.done:
    add   rsp, 16 
    pop   r15 
    pop   r14
    pop   r13
    pop   r12
    pop   rbx
    ret