; ------------------------------------------------------------
; void matvec(const float* W, const float* x, float* out,
;             size_t rows, size_t cols)
;   out[r] = dot(row r of W, x)   for r in 0..rows-1
;   W is row-major, rows x cols
;
; Args:  rdi = W   rsi = x   rdx = out   rcx = rows   r8 = cols
; Ret:   nothing
; Calls: dot
; Saved: rbx, r12-r15 (callee-saved, pushed/popped)
;   r12 = current row of W    r13 = x     r14 = current out slot
;   r15 = rows remaining      rbx = cols
; Stack: 5 pushes + return address = 48 bytes -> aligned for call
; ------------------------------------------------------------
global matvec
extern dot

section .text
matvec:
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov r12, rdi ; r12 = W
    mov r13, rsi ; r13 = X
    mov r14, rdx ; r14 = out
    mov r15, rcx ; rcx = rows
    mov rbx, r8 ; rbx = cols

    test r15, r15 ; rows == 0?
    jz .done

.loop:
    mov rdi, r12 ; dot arg 1 : W
    mov rsi, r13 ; dot ar2 2 : X
    mov rdx, rbx ; dot arg 3 : cols
    call dot ; xmm0 = dot
    movss [r14], xmm0 ; store output of dot in out
    lea r12, [r12 + rbx*4] ; next row 
    add r14, 4 ; next out slot
    dec r15 ; rows--
    jnz .loop

.done:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret