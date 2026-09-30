; ------------------------------------------------------------
; float add_f(float a, float b)
;   returns a + b
;
; Args:  xmm0 = a   xmm1 = b
; Ret:   xmm0
; Uses:  nothing else
; ------------------------------------------------------------
global add_f
section .text
add_f:
    addss xmm0, xmm1 ; xmm0 = a + b
    ret