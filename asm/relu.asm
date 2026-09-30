; ------------------------------------------------------------
; float relu(float x)
;   returns max(0, x)
;
; Args:  xmm0 = x
; Ret:   xmm0
; Uses:  xmm1 (scratch, holds 0.0)
; ------------------------------------------------------------
global relu
section .text
relu:
    xorps xmm1, xmm1 ; xmm1 = 0.0
    maxss xmm0, xmm1 ; xmm0 = max(0.0, x)
    ret