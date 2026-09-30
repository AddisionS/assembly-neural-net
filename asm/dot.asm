; ------------------------------------------------------------
; float dot(const float* a, const float* b, size_t n)
;   returns sum of a[i] * b[i] for i in 0..n-1
;
; Args:  rdi = a   rsi = b   rdx = n
; Ret:   xmm0 (running sum)
; Uses:  xmm1 (scratch); rdi, rsi, rdx are modified
; ------------------------------------------------------------
global dot
section .text
dot:
    xorps xmm0, xmm0 ; sum = 0.0
    test rdx, rdx ; n == 0?
    jz .done ; empty -> return 0
.loop:
    movss xmm1, [rdi] ; xmm1 = a[i]
    mulss xmm1, [rsi] ; xmm1 = a[i] * b[i]
    addss xmm0, xmm1 ; sum += a[i] * b[i]
    add rdi, 4 ; a++
    add rsi, 4 ; b++
    dec rdx ; n--
    jnz .loop ; repeat while n != 0
.done: 
    ret