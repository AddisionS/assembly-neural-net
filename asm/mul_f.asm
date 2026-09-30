; ------------------------------------------------------------
; float mul_f(float a, float b)
;   returns a * b
;
; Args:  xmm0 = a   xmm1 = b
; Ret:   xmm0
; Uses:  nothing else
; ------------------------------------------------------------
global mul_f
section .text
mul_f:
    mulss xmm0, xmm1 ; xmm0 = a * b
    ret