; ------------------------------------------------------------
; float relu_grad(float x)
;   returns 1.0 if x > 0, else 0.0   (derivative of ReLU)
;
; Args:  xmm0 = x
; Ret:   xmm0
; Uses:  xmm1 (mask), xmm2 (holds 1.0)
; ------------------------------------------------------------
global relu_grad

section .rodata
one : dd 1.0 ; constant 1.0f

section .text
relu_grad:
    xorps xmm1, xmm1 ; xmm1 = 0.0
    cmpltss xmm1, xmm0 ; mask = (0.0 < x) ? all 1 : all 0
    movss xmm2, [rel one] ; xmm2 = 1.0
    andps xmm1, xmm2 ; mask & 1.0 -> 1.0 or 0.0
    movaps xmm0, xmm1 ; move result into return register
    ret