; ------------------------------------------------------------
; float sum_array(const float* p, size_t n)
;   returns p[0] + p[1] + ... + p[n-1]
;
; Args:  rdi = p   rsi = n
; Ret:   xmm0 (running sum)
; Uses:  rdi, rsi are modified (pointer and counter)
; ------------------------------------------------------------
global sum_array
section .text
sum_array:
    xorps xmm0, xmm0 ; sum = 0.0
    test rsi, rsi ; n == 0?
    jz .done ; empty array -> return 0
.loop:
    addss xmm0, [rdi] ; sum += *p
    add rdi, 4 ; p++
    dec rsi ; n--
    jnz .loop ; repeat while n != 0
.done:
    ret