; ------------------------------------------------------------
; void relu_array(float* x, size_t n)
;   x[i] = max(0, x[i])   for i in 0..n-1   (in place)
;
; Args:  rdi = x   rsi = n
; Ret:   nothing
; Uses:  xmm0 (scratch), xmm1 (holds 0.0 for the whole loop)
;        rdi, rsi are modified
; ------------------------------------------------------------
global relu_array
section .text
relu_array:
    test rsi, rsi ; n == 0 ?
    jz .done ; nothing to do
    xorps xmm1, xmm1 ; xmm1 = 0.0
.loop:
    movss xmm0, [rdi] ; xmm0 = x[i]
    maxss xmm0, xmm1 ; xmm0 = max(x[i], 0.0)
    movss [rdi], xmm0 ; x[i] = xmm0
    add rdi, 4 ; x++
    dec rsi ; n--
    jnz .loop ; repeat while n != 0
.done:
    ret