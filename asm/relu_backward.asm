; ------------------------------------------------------------
; void relu_backward(float* delta, const float* z, size_t n)
;   delta[i] = (z[i] > 0) ? delta[i] : 0     (in place)
;
; Args:  rdi = delta   rsi = z   rdx = n
; Ret:   nothing
; Uses:  xmm0, xmm1, xmm2; rdi, rsi, rdx are modified
; ------------------------------------------------------------
global relu_backward
section .text
relu_backward:
    test rdx, rdx ; n == 0?
    jz .done 
    xorps xmm0, xmm0 ; xmm0 = 0
.loop:
    movaps xmm1, xmm0 ; xmm1 = 0.0
    movss xmm2, [rsi] ; xmm2 = z[i]
    cmpltss xmm1, xmm2 ; mask = (0 < z[i]) ? all 1s : 0s
    movss xmm2, [rdi] ; xmm2 = delta[i]
    andps xmm2, xmm1 ; keep delta only where the mask is on 
    movss [rdi], xmm2 ; delta[i] = result
    add rdi, 4 ; delta++
    add rsi, 4 ; z ++
    dec rdx ; n--
    jnz .loop
.done:
    ret