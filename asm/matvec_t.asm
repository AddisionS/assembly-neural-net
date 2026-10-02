; ------------------------------------------------------------
; void matvec_t(const float* W, const float* v, float* out,
;               size_t rows, size_t cols)
;   out[c] = sum over r of  W[r][c] * v[r]       (out has `cols` elements)
;   W is row-major, rows x cols. v has `rows` elements.
;
; Args:  rdi = W   rsi = v   rdx = out   rcx = rows   r8 = cols
; Ret:   nothing
; Calls: axpy
; Saved: r12 = current row of W   r13 = v pointer   r14 = out
;        r15 = rows remaining     rbx = cols
; Stack: 5 pushes + return address = 48 bytes -> aligned for call
; ------------------------------------------------------------
global matvec_t
extern axpy

section .text
matvec_t:
    push  rbx
    push  r12
    push  r13
    push  r14
    push  r15

    mov   r12, rdi ; r12 = W (current row)
    mov   r13, rsi ; r13 = v pointer
    mov   r14, rdx ; r14 = out
    mov   r15, rcx ; r15 = rows remaining
    mov   rbx, r8 ; rbx = cols

    ; ---- zero out[0..cols-1] ----
    test  rbx, rbx ; cols == 0 ?
    jz    .done
    xorps xmm1, xmm1 ; xmm1 = 0.0
    mov   rax, r14 ; walker pointer
    mov   r9,  rbx ; counter = cols
.zero:
    movss [rax], xmm1 ; *p = 0
    add   rax, 4
    dec   r9
    jnz   .zero

    ; ---- out += v[r] * row r, for every row ----
    test  r15, r15 ; rows == 0 ?
    jz    .done
.loop:
    movss xmm0, [r13] ; alpha = v[r]
    mov   rdi, r12 ; axpy arg: x = current row of W
    mov   rsi, r14 ; axpy arg: y = out
    mov   rdx, rbx ; axpy arg: n = cols
    call  axpy ; out += v[r] * row
    lea   r12, [r12 + rbx*4] ; next row of W
    add   r13, 4 ; next v
    dec   r15
    jnz   .loop
.done:
    pop   r15 ; restore in REVERSE order
    pop   r14
    pop   r13
    pop   r12
    pop   rbx
    ret