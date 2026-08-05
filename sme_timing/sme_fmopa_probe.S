// AArch64 SME FMOPA timing loops. The caller is non-streaming.

        .text
        .p2align 4

        .globl sme_streaming_vector_bytes
        .type sme_streaming_vector_bytes, %function
sme_streaming_vector_bytes:
        smstart
        rdsvl   x0, #1
        smstop
        ret
        .size sme_streaming_vector_bytes, .-sme_streaming_vector_bytes

        .macro probe_begin
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        mov     x3, x1
        mov     x4, x2
        smstart za
        ptrue   p0.s
        isb
        mrs     x5, cntvct_el0
        .endm

        .macro probe_end
        isb
        mrs     x6, cntvct_el0
        mrs     x7, cntfrq_el0
        smstop
        sub     x6, x6, x5
        str     x6, [x3]
        str     x7, [x4]
        ldp     x29, x30, [sp], #16
        ret
        .endm

        .globl sme_measure_loop_baseline
        .type sme_measure_loop_baseline, %function
sme_measure_loop_baseline:
        probe_begin
.Lbaseline_loop:
        subs    x0, x0, #1
        b.ne    .Lbaseline_loop
        probe_end
        .size sme_measure_loop_baseline, .-sme_measure_loop_baseline

        .globl sme_measure_fmopa_one_tile
        .type sme_measure_fmopa_one_tile, %function
sme_measure_fmopa_one_tile:
        probe_begin
.Lone_tile_loop:
        fmopa   za0.s, p0/m, p0/m, z0.s, z1.s
        subs    x0, x0, #1
        b.ne    .Lone_tile_loop
        probe_end
        .size sme_measure_fmopa_one_tile, .-sme_measure_fmopa_one_tile

        .globl sme_measure_fmopa_four_tiles
        .type sme_measure_fmopa_four_tiles, %function
sme_measure_fmopa_four_tiles:
        probe_begin
.Lfour_tile_loop:
        fmopa   za0.s, p0/m, p0/m, z0.s, z1.s
        fmopa   za1.s, p0/m, p0/m, z2.s, z3.s
        fmopa   za2.s, p0/m, p0/m, z4.s, z5.s
        fmopa   za3.s, p0/m, p0/m, z6.s, z7.s
        subs    x0, x0, #1
        b.ne    .Lfour_tile_loop
        probe_end
        .size sme_measure_fmopa_four_tiles, .-sme_measure_fmopa_four_tiles

        .globl sme_run_loop_baseline
        .type sme_run_loop_baseline, %function
sme_run_loop_baseline:
        smstart za
        ptrue   p0.s
.Lclock_baseline_loop:
        subs    x0, x0, #1
        b.ne    .Lclock_baseline_loop
        smstop
        ret
        .size sme_run_loop_baseline, .-sme_run_loop_baseline

        .globl sme_run_fmopa_one_tile
        .type sme_run_fmopa_one_tile, %function
sme_run_fmopa_one_tile:
        smstart za
        ptrue   p0.s
.Lclock_one_tile_loop:
        fmopa   za0.s, p0/m, p0/m, z0.s, z1.s
        subs    x0, x0, #1
        b.ne    .Lclock_one_tile_loop
        smstop
        ret
        .size sme_run_fmopa_one_tile, .-sme_run_fmopa_one_tile

        .globl sme_run_fmopa_four_tiles
        .type sme_run_fmopa_four_tiles, %function
sme_run_fmopa_four_tiles:
        smstart za
        ptrue   p0.s
.Lclock_four_tile_loop:
        fmopa   za0.s, p0/m, p0/m, z0.s, z1.s
        fmopa   za1.s, p0/m, p0/m, z2.s, z3.s
        fmopa   za2.s, p0/m, p0/m, z4.s, z5.s
        fmopa   za3.s, p0/m, p0/m, z6.s, z7.s
        subs    x0, x0, #1
        b.ne    .Lclock_four_tile_loop
        smstop
        ret
        .size sme_run_fmopa_four_tiles, .-sme_run_fmopa_four_tiles

        .section .note.GNU-stack,"",%progbits
