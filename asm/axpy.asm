; ------------------------------------------------------------
; void axpy(float alpha, const float* x, float* y, size_t n)
;   y[i] += alpha * x[i]   for i in 0..n-1
;
; Args:  xmm0 = alpha   rdi = x   rsi = y   rdx = n
; Uses:  xmm1 (scratch)     Preserves: everything else
; ------------------------------------------------------------
global axpy
section .text
axpy:
    test rdx, rdx ; n == 0?
    jz .done ; nothing to do
.loop:
    movss xmm1, [rdi] ; xmm1 = x[i]
    mulss xmm1, xmm0  ; xmm1 = alpha * x[i]
    addss xmm1, [rsi] ; xmm1 += y[i]
    movss [rsi], xmm1 ; y[i] = xmm1
    add rdi, 4 ; x++
    add rsi, 4 ; y++
    dec rdx    ; n--
    jnz .loop  ; repeat while n != 0
.done:
    ret