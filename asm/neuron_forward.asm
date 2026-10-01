; ------------------------------------------------------------
; float neuron_forward(const float* w, const float* x, float b, size_t n)
;   returns relu(dot(w, x, n) + b)
;
; Args:  rdi = w   rsi = x   xmm0 = b   rdx = n
; Ret:   xmm0
; Calls: dot, relu
; Stack: 8 bytes to save b across the call to dot
;        (also re-aligns rsp to 16 bytes for the calls)
; ------------------------------------------------------------
global neuron_forward
extern dot
extern relu

section .text
neuron_forward:
    sub rsp, 8 ; reserve 8 byte on stack
    movss [rsp], xmm0 ; move b to stack
    call dot ; xmm0 = dot(w, x, n)
    addss xmm0, [rsp] ; add b to xmm0
    call relu ; xmm0 = max(0, xmm0)
    add rsp, 8 ; give space to stack
    ret