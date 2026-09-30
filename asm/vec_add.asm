; ------------------------------------------------------------
; void vec_add(const float* a, const float* b, float* out, size_t n)
;   out[i] = a[i] + b[i]   for i in 0..n-1
;   (out may be the same array as a or b)
;
; Args:  rdi = a   rsi = b   rdx = out   rcx = n
; Ret:   nothing
; Uses:  xmm0 (scratch); rdi, rsi, rdx, rcx are modified
; ------------------------------------------------------------
global vec_add
section .text
vec_add:
    test rcx, rcx ; n == 0?
    jz .done ; nothing to do
.loop:
    movss xmm0, [rdi] ; xmm0 = a[i]
    movss xmm1, [rsi] ; xmm1 = b[i]
    addss xmm0, xmm1 ; xmm0 = a[i] + b[i]
    movss [rdx], xmm0 ; out[i] = xmm0
    add rdi, 4 ; a++
    add rsi, 4 ; b++
    add rdx, 4 ; out++
    dec rcx ; n--
    jnz .loop ; repeat while n != 0
.done:
    ret