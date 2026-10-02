; ------------------------------------------------------------
; float exp_f(float x)
;   returns e^x   (x is clamped to [-87, 88])
;
;   x = n*ln2 + g,  e^x = 2^n * e^g
;   e^g by 6th-order Taylor polynomial (Horner), 2^n by exponent bits
;
; Args:  xmm0 = x
; Ret:   xmm0
; Uses:  xmm1, xmm2, eax
; ------------------------------------------------------------
global exp_f

section .rodata
lo:     dd -87.0
hi:     dd 88.0
log2e:  dd 1.4426950409
ln2:    dd 0.6931471806
c6:     dd 0.0013888889 ; 1/720
c5:     dd 0.0083333333 ; 1/120
c4:     dd 0.0416666667 ; 1/24
c3:     dd 0.1666666667 ; 1/6
c2:     dd 0.5 ; 1/2
one:    dd 1.0 ; 1/1 and 1/1

section .text
exp_f:
    maxss xmm0, [rel lo] ; x = max(x , -87)
    minss xmm0, [rel hi] ; x = min(x , 87)

    movaps   xmm1, xmm0
    mulss    xmm1, [rel log2e] ; xmm1 = x / ln2
    cvtss2si eax, xmm1 ; eax = n = round(x / ln2)
    cvtsi2ss xmm2, eax ; xmm2 = (float) n
    mulss    xmm2, [rel ln2] ; xmm2 = n * ln2
    subss    xmm0, xmm2 ; xmm0 = g = x - n*ln2   (small)

    ; Horner: e^g = ((((((c6*g + c5)*g + c4)*g + c3)*g + c2)*g + 1)*g + 1)
    movss    xmm1, [rel c6]
    mulss    xmm1, xmm0
    addss    xmm1, [rel c5]
    mulss    xmm1, xmm0
    addss    xmm1, [rel c4]
    mulss    xmm1, xmm0
    addss    xmm1, [rel c3]
    mulss    xmm1, xmm0
    addss    xmm1, [rel c2]
    mulss    xmm1, xmm0
    addss    xmm1, [rel one]
    mulss    xmm1, xmm0
    addss    xmm1, [rel one] ; xmm1 = e^g

    add      eax, 127 ; biased exponent = n + 127
    shl      eax, 23 ; move it into the float's exponent bits
    movd     xmm2, eax ; xmm2 = the float 2^n
    mulss    xmm1, xmm2 ; e^x = e^g * 2^n
    movaps   xmm0, xmm1 ; return in xmm0
    ret