{%- set param_outer_len = 'qword ptr [rdi]' %}
{%- set param_middle_lm = 'qword ptr [rdi + 0x08]' %}
{%- set param_outer_matrix = 'qword ptr [rdi + 0x10]' %}
{%- set param_middle_matrix = 'qword ptr [rdi + 0x18]' %}
{%- set param_inner_matrix = 'qword ptr [rdi + 0x20]' %}
{%- set param_outer_stride_0 = 'qword ptr [rdi + 0x28]' %}
{%- set param_outer_stride_1 = 'qword ptr [rdi + 0x30]' %}
{%- set param_middle_stride_0 = 'qword ptr [rdi + 0x38]' %}
{%- set param_middle_stride_1 = 'qword ptr [rdi + 0x40]' %}
{%- set param_inner_stride_0 = 'qword ptr [rdi + 0x48]' %}
{%- set param_inner_stride_1 = 'qword ptr [rdi + 0x50]' %}
{%- set param_flags = 'byte ptr [rdi + 0x58]' %}

{%- set F_OUTER_IS_CONTIG = '0b001' %}
{%- set F_MIDDLE_IS_CONTIG = '0b010' %}
{%- set F_OVERWRITE_C = '0b100' %}

{%- set ARCH_R = DIM_R // 8 %}

// DIM_R: {{DIM_R}}
// DIM_S: {{DIM_S}}
// LOOP_ORDER: {{LOOP_ORDER}}

{%- macro outer_move_generic(label, is_overwrite) %}
    {#- When loop order is CAB or CBA, we store the outer matrix here instead of load #}

    mov r14, rax
    mov r10, rdx
    mov r8, {{param_outer_stride_1}}
    mov r13, {{param_outer_stride_0}}

    {%- for i in range(0, DIM_S) %}
.{{label}}_next_s{{i}}:
        {%- if i != 0 %}
            add r14, r8
        {%- endif %}
        mov r11, r10
        mov r15, r14
        {%- for j in range(0, ARCH_R) %}
            {%- set elem_base = j * simd_elems %}

            {%- for a_i in range(0, 2) %}
                {%- if LOOP_ORDER == "cab" or LOOP_ORDER == "cba" %}
                    {#- Prepare for store #}

                    vextractf128 xmm15, ymm{{i * ARCH_R + j}}, {{a_i}}
                {%- endif %}

                call .{{label}}_xmm15

                {%- if not (LOOP_ORDER == "cab" or LOOP_ORDER == "cba") %}
                    {#- Finish load #}

                    vinsertf128 ymm{{i * ARCH_R + j}}, ymm{{i * ARCH_R + j}}, xmm15, {{a_i}}
                {%- endif %}

                {%- if not (a_i == 2-1 and j == ARCH_R-1) %}
                    cmp r11, 0
                    je .{{label}}_next_s{{i + 1}}
                {%- endif %}
            {%- endfor %}
        {%- endfor %}
    {%- endfor %}
    jmp .{{label}}_next_s{{DIM_S}}

.{{label}}_xmm15:
    {%- for a_j in range(0, 4) %}
        {%- if LOOP_ORDER == "cab" or LOOP_ORDER == "cba" %}
            {#- Store #}

            {%- if is_overwrite %}
                vextractps [r15], xmm15, {{a_j}}
            {%- else %}
                vextractps r9d, xmm15, {{a_j}}
                vmovd xmm14, r9d
                vaddps xmm14, xmm14, [r15]
                vmovss [r15], xmm14
            {%- endif %}

        {%- else %}
            {#- Load #}

            {%- if a_j == 0 %}
                vmovss xmm15, [r15]
            {%- else %}
                vinsertps xmm15, xmm15, [r15], {{a_j}} << 4
            {%- endif %}

        {%- endif %}

        add r15, r13
        dec r11
        {%- if a_j != 4-1 %}
            jz .{{label}}_xmm15_ret
        {%- endif %}
    {%- endfor %}
.{{label}}_xmm15_ret:
    ret

.{{label}}_next_s{{DIM_S}}:
{%- endmacro %}

{%- macro outer_move_contig(label, is_overwrite=false) %}
    {#- When loop order is CAB or CBA, we store the outer matrix here instead of load #}

    mov r14, rax
    mov r8, {{param_outer_stride_1}}

    {%- set startreg = ARCH_R * DIM_S %}
    {%- for i in range(0, DIM_S) %}
        {%- for j in range(0, ARCH_R) %}

            {%- if LOOP_ORDER == "cab" or LOOP_ORDER == "cba" %}
                {#- Store #}

                {%- if is_overwrite %}
                    vmovaps [r14 + {{j * simd_bytes}}], ymm{{i * ARCH_R + j}}
                {%- else %}
                    vaddps ymm{{startreg + j}}, ymm{{i * ARCH_R + j}}, [r14 + {{j * simd_bytes}}]
                    vmovaps [r14 + {{j * simd_bytes}}], ymm{{startreg + j}}
                {%- endif %}
            {%- else %}
                {#- Load #}

                vmovaps ymm{{i * ARCH_R + j}}, [r14 + {{j * simd_bytes}}]
            {%- endif %}

        {%- endfor %}

        {%- if i != DIM_S-1 %}
            add r14, r8
        {%- endif %}
    {%- endfor %}
{%- endmacro %}

.intel_syntax noprefix
.text
.p2align 5
.globl mmkernel_avx2_sss_{{DIM_R}}x{{DIM_S}}_{{LOOP_ORDER}}_{{SUFFIX}}
mmkernel_avx2_sss_{{DIM_R}}x{{DIM_S}}_{{LOOP_ORDER}}_{{SUFFIX}}:
.cfi_startproc

    push        rbp
    mov         rbp, rsp

    push        rbx
    push        r12
    push        r13
    push        r14
    push        r15
.cfi_def_cfa_offset 48

    stmxcsr     [rsp + 4]
    mov         rbx, 0x1FC0
    mov         [rsp], ebx
    ldmxcsr     [rsp]

    vzeroall

{%- set scratch_reg_i_start = ARCH_R * DIM_S %}

{%- set elem_size = 4 %}
{%- set simd_bits = 256 %}
{%- set simd_bytes = simd_bits // 8 %}
{%- set simd_elems = simd_bits // 8 // elem_size %}

{#-
Registers:
- rdi: args
- rsi: outer_len
- rax: outer_matrix
- rbx: middle_matrix
- rcx: inner_matrix
- rdx: middle_lm
- r10: middle_stride_0
- r11: middle_stride_1
- r12: inner_stride_0
- r13: inner_stride_1

- r14
- r15
- r8
- r9
#}

    mov rsi, {{param_outer_len}}
    mov rdx, {{param_middle_lm}}
    mov rax, {{param_outer_matrix}}
    mov rbx, {{param_middle_matrix}}
    mov rcx, {{param_inner_matrix}}

{%- if not (LOOP_ORDER == "cab" or LOOP_ORDER == "cba") %}
    {#- Load outer A/B matrix (RxS elems) #}

    test {{param_flags}}, {{F_OUTER_IS_CONTIG}}
    jz .outer_move_generic

.outer_move_contig:
    {{ outer_move_contig("outer_move_contig") }}
    jmp .o_loop_before

.outer_move_generic:
    {{ outer_move_generic("outer_move_generic") }}

.o_loop_before:

{%- endif %}

    {#- some of them are used in outer_move_generic #}
    mov r10, {{param_middle_stride_0}}
    mov r11, {{param_middle_stride_1}}
    mov r12, {{param_inner_stride_0}}
    mov r13, {{param_inner_stride_1}}

{#- Outer loop #}
.o_loop:

    {%- if LOOP_ORDER == "acb" or LOOP_ORDER == "bca" %}
        {#- Zero mid matrix, which is C #}
        
        {%- for j in range(0, ARCH_R) %}
            vxorps ymm{{scratch_reg_i_start + j}}, ymm{{scratch_reg_i_start + j}}, ymm{{scratch_reg_i_start + j}}
        {%- endfor %}

    {%- else %}
        {#- Load mid A/B matrix (R elems) strided by middle_stride_0 #}

        test {{param_flags}}, {{F_MIDDLE_IS_CONTIG}}
        jz .move_middle_generic

        {#- Contiguous load #}
        {%- for j in range(0, ARCH_R) %}
            vmovaps ymm{{scratch_reg_i_start + j}}, [rbx + {{j * simd_bytes}}]
        {%- endfor %}
    .move_middle_generic_r:

    {%- endif %}

    {#- Middle loop #}
    mov r15, rcx
    {%- for i_mid in range(0, DIM_S) %}

        {#- Load A/B (1 elem) strided by stride_inner #}
        {%- if LOOP_ORDER == "abc" or LOOP_ORDER == "bac" %}
            vxorps ymm15, ymm15, ymm15
            {%- for i_inn in range(0, ARCH_R) %}
                vfmadd231ps ymm15, ymm{{i_mid * ARCH_R + i_inn}}, ymm{{scratch_reg_i_start + i_inn}}
            {%- endfor %}

            {#- Horizontal sum of ymm15 #}
            {%- if DIM_S * ARCH_R + ARCH_R + 1 > 15 and i_mid != DIM_S-1 %}
                vmovups [rsp - 32], ymm14 {#- that's a bit sad #}
            {%- endif %}
            vextractf128 xmm14, ymm15, 1
            vaddps xmm15, xmm14, xmm15
            vmovhlps xmm14, xmm15, xmm15
            vaddps xmm15, xmm15, xmm14
            vshufps xmm14, xmm15, xmm15, 1
            vaddss xmm15, xmm15, xmm14
            {%- if DIM_S * ARCH_R + ARCH_R + 1 > 15 and i_mid != DIM_S-1 %}
                vmovups ymm14, [rsp - 32]
            {%- endif %}
            
            vaddss xmm15, xmm15, [r15]
            vmovss [r15], xmm15
            {#- C = C + partial result #}

        {%- else %}
            vbroadcastss ymm15, dword ptr [r15]

            {#- Inner loop #}
            {%- for i_inn in range(0, ARCH_R) %}

                {%- if LOOP_ORDER == "cab" or LOOP_ORDER == "cba" %}
                    {#- OUTER += MID * INNER #}
                    vfmadd231ps ymm{{i_mid * ARCH_R + i_inn}}, ymm{{scratch_reg_i_start + i_inn}}, ymm15
                {%- else %}
                    {#- MID += OUTER * INNER #}
                    vfmadd231ps ymm{{scratch_reg_i_start + i_inn}}, ymm{{i_mid * ARCH_R + i_inn}}, ymm15
                {%- endif %}
            {%- endfor %}
        {%- endif %}

        {%- if i_mid != DIM_S %}
            add r15, r12
        {%- endif %}
    {%- endfor %}

{%- if LOOP_ORDER == "acb" or LOOP_ORDER == "bca" %}
    {#- Save mid C matrix (R elems) strided by middle_stride_0 #}

    test {{param_flags}}, {{F_MIDDLE_IS_CONTIG}}
    jz .move_middle_generic

    {#- Contiguous #}
    {%- for j in range(0, ARCH_R) %}
        vmovaps [rbx + {{j * simd_bytes}}], ymm{{scratch_reg_i_start + j}}
    {%- endfor %}
.move_middle_generic_r:

{%- endif %}

    add rbx, r11 {#- middle_matrix += middle_stride_1 #}
    add rcx, r13 {#- inner_matrix += inner_stride_1 #}
    dec rsi
    jnz .o_loop

{%- if LOOP_ORDER == "cab" or LOOP_ORDER == "cba" %}
    {#- Store-accumulate outer C matrix (RxS elems) #}

    test {{param_flags}}, {{F_OUTER_IS_CONTIG}}
    jz .outer_move_generic

    test {{param_flags}}, {{F_OVERWRITE_C}}
    jnz .outer_move_overwrite_contig

.outer_move_contig:
    {{ outer_move_contig("outer_move_contig", is_overwrite=false) }}
    jmp .return

.outer_move_overwrite_contig:
    {{ outer_move_contig("outer_move_overwrite_contig", is_overwrite=true) }}
    jmp .return

.outer_move_generic:
    test {{param_flags}}, {{F_OVERWRITE_C}}
    jnz .outer_move_overwrite_generic

.outer_move_generic_acc:
    {{ outer_move_generic("outer_move_generic_acc", is_overwrite=false) }}
    jmp .return

.outer_move_overwrite_generic:
    {{ outer_move_generic("outer_move_overwrite_generic", is_overwrite=true) }}

{%- endif %}

.return:
    ldmxcsr     [rsp + 4]

    mov rax, 0

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx

    mov rsp, rbp
    pop rbp
    ret

.move_middle_generic:
    {#- When loop order is ACB or BCA, we store the mid matrix here instead of load #}
    mov r14, rdx
    sub r14, {{simd_elems * (ARCH_R - 1)}}
    mov r15, rbx

    {%- if DIM_S * ARCH_R + ARCH_R + 1 > 15 and LOOP_ORDER == "acb" or LOOP_ORDER == "bca" %}
        vmovups [rsp - 32], ymm14 {#- that's a bit sad #}
    {%- endif %}

    {%- for j in range(0, ARCH_R) %}
        {%- if j == ARCH_R - 1 %}
            cmp r14, {{simd_elems}}
            jl .move_middle_generic_border
        {%- endif %}

        {%- for a_i in range(0, 2) %}
            {%- if LOOP_ORDER == "acb" or LOOP_ORDER == "bca" %}
                {%- if scratch_reg_i_start + j == 14 %}
                    vmovups ymm14, [rsp - 32]
                {%- endif %}
                vextractf128 xmm15, ymm{{scratch_reg_i_start + j}}, {{a_i}}
            {%- endif %}

            {%- for a_j in range(0, 4) %}
                {%- set is_last_el = a_i == 2-1 and a_j == 4-1 and j == ARCH_R-1 %}

                {%- if LOOP_ORDER == "acb" or LOOP_ORDER == "bca" %}
                    vextractps r9d, xmm15, {{a_j}}
                    vmovd xmm14, r9d
                    vaddps xmm14, xmm14, [r15]
                    vmovss [r15], xmm14
                {%- else %}
                    {%- if a_j == 0 %}
                        vmovss xmm15, [r15]
                    {%- else %}
                        vinsertps xmm15, xmm15, [r15], {{a_j}} << 4
                    {%- endif %}
                {%- endif %}

                {%- if not is_last_el %}
                    add r15, r10
                {%- endif %}
            {%- endfor %}

            {%- if not (LOOP_ORDER == "acb" or LOOP_ORDER == "bca") %}
                vinsertf128 ymm{{scratch_reg_i_start + j}}, ymm{{scratch_reg_i_start + j}}, xmm15, {{a_i}}
            {%- endif %}
        {%- endfor %}
    {%- endfor %}

    jmp .move_middle_generic_r

.move_middle_generic_border:
    {%- set j = ARCH_R - 1 %}
    {%- for a_i in range(0, 2) %}
        {%- if LOOP_ORDER == "acb" or LOOP_ORDER == "bca" %}
            vextractf128 xmm15, ymm{{scratch_reg_i_start + j}}, {{a_i}}
        {%- endif %}

        {%- for a_j in range(0, 4) %}
            {%- set is_last_el = a_i == 2-1 and a_j == 4-1 %}

            {%- if LOOP_ORDER == "acb" or LOOP_ORDER == "bca" %}
                vextractps r9d, xmm15, {{a_j}}
                vmovd xmm14, r9d
                vaddps xmm14, xmm14, [r15]
                vmovss [r15], xmm14
            {%- else %}
                {%- if a_j == 0 %}
                    vmovss xmm15, [r15]
                {%- else %}
                    vinsertps xmm15, xmm15, [r15], {{a_j}} << 4
                {%- endif %}
            {%- endif %}

            {%- if not is_last_el %}
                dec r14
                {%- if not (LOOP_ORDER == "acb" or LOOP_ORDER == "bca") %}
                    jz .move_middle_generic_border_insert_{{a_i}}
                {%- else %}
                    jz .move_middle_generic_r
                {%- endif %}
                add r15, r10
            {%- endif %}
        {%- endfor %}
        {%- if not (LOOP_ORDER == "acb" or LOOP_ORDER == "bca") %}
            {%- if a_i == 1 %}
.move_middle_generic_border_insert_1:
            {%- endif %}
            vinsertf128 ymm{{scratch_reg_i_start + j}}, ymm{{scratch_reg_i_start + j}}, xmm15, {{a_i}}
        {%- endif %}
    {%- endfor %}
    jmp .move_middle_generic_r
    {%- if not (LOOP_ORDER == "acb" or LOOP_ORDER == "bca") %}
.move_middle_generic_border_insert_0:
        vinsertf128 ymm{{scratch_reg_i_start + j}}, ymm{{scratch_reg_i_start + j}}, xmm15, 0
        jmp .move_middle_generic_r
    {%- endif %}


.cfi_endproc


