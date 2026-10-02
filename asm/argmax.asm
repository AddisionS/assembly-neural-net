; ------------------------------------------------------------
; size_t argmax(const float* x, size_t n)
;   returns the index of the largest element (first one on ties)
;
; Args:  rdi = x   rsi = n
; Ret:   rax = index (0 if n == 0)
; Uses:  xmm0 = best value so far, xmm1 = x[i], rcx = i
; ------------------------------------------------------------
global argmax
section .text
argmax:
    xor eax, eax ; best index = 0
    test rsi, rsi ; n == 0 ?
    jz .done
    movss xmm0, [rdi] ; xmm0 = x[0]
    mov rcx, 1 ; i = i
.loop:
    cmp rcx, rsi ; i >= n ?
    jae .done 
    movss xmm1, [rdi + rcx * 4] ; xmm1 = x[i]
    comiss xmm1, xmm0 ; compare x[i] with best
    jbe .next ; keep old one
    movaps xmm0, xmm1 ; best = x[i]
    mov rax, rcx ; best index = i
.next:
    inc rcx ; i++
    jmp .loop
.done:
    ret