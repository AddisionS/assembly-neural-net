; ------------------------------------------------------------
; void softmax(const float* in, float* out, size_t n)
;   out[i] = exp(in[i] - max) / sum_j exp(in[j] - max)
;
; Args:  rdi = in   rsi = out   rdx = n
; Ret:   nothing
; Calls: exp_f
; Saved: rbx = in ptr   r12 = out ptr   r13 = counter
;        r14 = out start   r15 = n
; Stack: 5 pushes + 16 bytes ([rsp] = max, [rsp+4] = sum)
;        5 pushes + return address = 48 -> aligned; 16 keeps it aligned
; ------------------------------------------------------------
global softmax
extern exp_f

section .text
softmax:
    push  rbx
    push  r12
    push  r13
    push  r14
    push  r15
    sub   rsp, 16

    mov   rbx, rdi ; rbx = in
    mov   r12, rsi ; r12 = out
    mov   r14, rsi ; r14 = out start (for pass 3)
    mov   r15, rdx ; r15 = n
    test  r15, r15 ; n == 0 ?
    jz    .done

    ; ---- pass 1: find the max ----
    movss xmm0, [rbx] ; max = in[0]
    mov   rax, rbx ; walker pointer
    mov   r13, r15 ; counter = n
.max_loop:
    movss xmm1, [rax]
    maxss xmm0, xmm1 ; max = max(max, in[i])
    add   rax, 4
    dec   r13
    jnz   .max_loop
    movss [rsp], xmm0 ; save max on the stack

    ; ---- pass 2: out[i] = exp_f(in[i] - max), sum += out[i] ----
    xorps xmm0, xmm0
    movss [rsp+4], xmm0 ; sum = 0
    mov   r13, r15 ; counter = n
.exp_loop:
    movss xmm0, [rbx] ; xmm0 = in[i]
    subss xmm0, [rsp] ; xmm0 = in[i] - max
    call  exp_f ; xmm0 = e^xmm0
    movss [r12], xmm0 ; out[i] = result
    movss xmm1, [rsp+4] ; reload sum (the call trashed xmm regs)
    addss xmm1, xmm0
    movss [rsp+4], xmm1 ; sum += out[i]
    add   rbx, 4
    add   r12, 4
    dec   r13
    jnz   .exp_loop

    ; ---- pass 3: divide everything by the sum ----
    movss xmm1, [rsp+4] ; xmm1 = sum
    mov   rax, r14 ; back to the start of out
    mov   r13, r15 ; counter = n
.div_loop:
    movss xmm0, [rax]
    divss xmm0, xmm1 ; out[i] /= sum
    movss [rax], xmm0
    add   rax, 4
    dec   r13
    jnz   .div_loop

.done:
    add   rsp, 16
    pop   r15               
    pop   r14
    pop   r13
    pop   r12
    pop   rbx
    ret