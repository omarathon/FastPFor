#!/usr/bin/env python3
"""
Generate simdunalignedbitpacking_u16.cpp — SIMD unaligned bit-packing for uint16,
with fused sum aggregation (no output writes during unpack).

Supports 128-bit (AVX) and 256-bit (AVX2) register widths.

Usage:
  python gen_simdunalignedbitpacking_u16.py --width 128   # AVX (128-bit registers)
  python gen_simdunalignedbitpacking_u16.py --width 256   # AVX2 (256-bit registers)
"""
import argparse
import os
import sys

LANE_BITS = 16
MAX_BIT = 16


def intrinsics(width):
    """Return a dict of intrinsic names for the given register width."""
    if width == 128:
        return dict(
            reg='__m128i',
            load='_mm_loadu_si128',
            store='_mm_storeu_si128',
            setzero='_mm_setzero_si128',
            or_fn='_mm_or_si128',
            and_fn='_mm_and_si128',
            slli='_mm_slli_epi16',
            srli='_mm_srli_epi16',
            set1='_mm_set1_epi16',
            add='_mm_add_epi32',
            add_u16='_mm_add_epi16',
            unpacklo='_mm_unpacklo_epi16',
            unpackhi='_mm_unpackhi_epi16',
        )
    else:
        return dict(
            reg='__m256i',
            load='_mm256_loadu_si256',
            store='_mm256_storeu_si256',
            setzero='_mm256_setzero_si256',
            or_fn='_mm256_or_si256',
            and_fn='_mm256_and_si256',
            slli='_mm256_slli_epi16',
            srli='_mm256_srli_epi16',
            set1='_mm256_set1_epi16',
            add='_mm256_add_epi32',
            add_u16='_mm256_add_epi16',
            unpacklo='_mm256_unpacklo_epi16',
            unpackhi='_mm256_unpackhi_epi16',
        )


def gen_header(I):
    # Width-specific helpers (256-bit AVX2 only — 128-bit path keeps the
    # narrower helpers below).
    if I['width'] == 256:
        delta_helpers = """\
// ── Delta+ZigZag helpers (AVX2, 16 uint16 lanes) ─────────────────────────────

// ZigZag decode: ((x >> 1) ^ -(x & 1)) per uint16 lane.
static inline __m256i zigzag_dec_u16_avx2(__m256i x) {
    __m256i odd = _mm256_and_si256(x, _mm256_set1_epi16(1));
    __m256i neg_odd = _mm256_sub_epi16(_mm256_setzero_si256(), odd);
    __m256i half = _mm256_srli_epi16(x, 1);
    return _mm256_xor_si256(half, neg_odd);
}

// Inclusive prefix sum over 16 uint16 lanes in a __m256i.
static inline __m256i prefix_sum_u16_avx2(__m256i x) {
    // 8-lane Sklansky prefix sum within each 128-bit half (3 levels).
    x = _mm256_add_epi16(x, _mm256_slli_si256(x, 2));
    x = _mm256_add_epi16(x, _mm256_slli_si256(x, 4));
    x = _mm256_add_epi16(x, _mm256_slli_si256(x, 8));
    // Bridge halves: broadcast lane 7 (top of low half) into all 8 lanes of
    // the high half; low half gets zero (via -1 byte indices in shuffle_epi8).
    __m256i lo_in_both = _mm256_permute2x128_si256(x, x, 0x00);
    const __m256i bcast_pattern = _mm256_setr_epi8(
        (char)0x80,(char)0x80,(char)0x80,(char)0x80,
        (char)0x80,(char)0x80,(char)0x80,(char)0x80,
        (char)0x80,(char)0x80,(char)0x80,(char)0x80,
        (char)0x80,(char)0x80,(char)0x80,(char)0x80,
        14, 15, 14, 15, 14, 15, 14, 15,
        14, 15, 14, 15, 14, 15, 14, 15);
    __m256i bcast = _mm256_shuffle_epi8(lo_in_both, bcast_pattern);
    return _mm256_add_epi16(x, bcast);
}

// Broadcast lane 15 (highest uint16 lane) of x across all 16 lanes.
static inline __m256i broadcast_lane15_u16_avx2(__m256i x) {
    __m256i hi_in_both = _mm256_permute2x128_si256(x, x, 0x11);
    const __m256i splat_pattern = _mm256_setr_epi8(
        14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
        14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15);
    return _mm256_shuffle_epi8(hi_in_both, splat_pattern);
}

// LOCAL pipeline: correction → zigzag_dec → per-OutReg prefix sum → aggregate.
// Lane 0 of OutReg is the per-OutReg anchor (zigzag-encoded as delta from 0).
static inline void agg_pipeline_local(__m256i OutReg, __m256i correction,
                                       __m256i* sum) {
    OutReg = _mm256_add_epi16(OutReg, correction);
    OutReg = zigzag_dec_u16_avx2(OutReg);
    OutReg = prefix_sum_u16_avx2(OutReg);
    *sum = _mm256_add_epi32(*sum, _mm256_unpacklo_epi16(OutReg, kZero));
    *sum = _mm256_add_epi32(*sum, _mm256_unpackhi_epi16(OutReg, kZero));
}

// CARRY pipeline: correction → zigzag_dec → prefix_sum → +carry → update carry
// (broadcast lane 15) → aggregate. carry holds prev OutReg's last decoded
// value broadcast across all 16 lanes.
static inline void agg_pipeline_carry(__m256i OutReg, __m256i correction,
                                       __m256i* carry, __m256i* sum) {
    OutReg = _mm256_add_epi16(OutReg, correction);
    OutReg = zigzag_dec_u16_avx2(OutReg);
    OutReg = prefix_sum_u16_avx2(OutReg);
    OutReg = _mm256_add_epi16(OutReg, *carry);
    *carry = broadcast_lane15_u16_avx2(OutReg);
    *sum = _mm256_add_epi32(*sum, _mm256_unpacklo_epi16(OutReg, kZero));
    *sum = _mm256_add_epi32(*sum, _mm256_unpackhi_epi16(OutReg, kZero));
}

"""
    else:
        # 128-bit path: delta variants not implemented — the codecs that use
        # them are AVX2-only.
        delta_helpers = ""

    return f"""\
/**
 * Auto-generated uint16 SIMD unaligned bit-packing with fused sum aggregation.
 * Generated by gen_simdunalignedbitpacking_u16.py (width={I['width']})
 *
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 */

#include "usimdbitpacking_u16.h"

namespace FastPForLib {{

namespace simdunaligned_u16 {{

static const {I['reg']} kZero = {I['setzero']}();

static inline void aggregate_sums_u16({I['reg']} OutReg, {I['reg']}* sum) {{
    // Zero-extend uint16 -> int32, then accumulate.
    *sum = {I['add']}(*sum, {I['unpacklo']}(OutReg, kZero));
    *sum = {I['add']}(*sum, {I['unpackhi']}(OutReg, kZero));
}}

// Corrected variant: fold per-OutReg exception correction into the aggregation.
// `correction` holds (exc - unpacked_gap) at exception lanes (mod 2^16) and 0
// elsewhere. uint16 add wraps the same way the codec encodes, so non-exception
// lanes are unchanged and exception lanes become exc.
static inline void aggregate_sums_u16_corrected({I['reg']} OutReg,
                                                 {I['reg']} correction,
                                                 {I['reg']}* sum) {{
    OutReg = {I['add_u16']}(OutReg, correction);
    *sum = {I['add']}(*sum, {I['unpacklo']}(OutReg, kZero));
    *sum = {I['add']}(*sum, {I['unpackhi']}(OutReg, kZero));
}}

{delta_helpers}static void SIMD_nullunpacker16(const {I['reg']} *__restrict__,
                                uint16_t *__restrict__) {{
}}

"""


def gen_pack_function(bit, I):
    """Generate pack function for given bit width."""
    lines = []
    lanes = I['lanes']
    block = I['block']
    total_inputs = block // lanes

    if bit == 0:
        lines.append(f"static void __SIMD_fastpackwithoutmask0_16("
                     f"const uint16_t *__restrict__, {I['reg']} *__restrict__) {{")
        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"static void __SIMD_fastpackwithoutmask{bit}_16("
                 f"const uint16_t *__restrict__ _in,")
    lines.append(f"    {I['reg']} *__restrict__ out) {{")
    lines.append(f"  const {I['reg']} *in = reinterpret_cast<const {I['reg']} *>(_in);")

    if bit == LANE_BITS:
        lines.append(f"  for (int i = 0; i < {total_inputs}; ++i) {{")
        lines.append(f"    {I['store']}(out + i, {I['load']}(in + i));")
        lines.append("  }")
        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"  {I['reg']} OutReg;")
    lines.append(f"  {I['reg']} InReg = {I['load']}(in);")
    lines.append("")

    bit_pos = 0
    lines.append("  OutReg = InReg;")
    bit_pos = bit

    for i in range(1, total_inputs):
        lines.append(f"  InReg = {I['load']}(++in);")
        remaining = LANE_BITS - bit_pos

        if remaining >= bit:
            if bit_pos == 0:
                lines.append("  OutReg = InReg;")
            else:
                lines.append(f"  OutReg = {I['or_fn']}(OutReg, {I['slli']}(InReg, {bit_pos}));")
            bit_pos += bit
        else:
            if remaining > 0:
                lines.append(f"  OutReg = {I['or_fn']}(OutReg, {I['slli']}(InReg, {bit_pos}));")
            lines.append(f"  {I['store']}(out++, OutReg);")
            if remaining > 0:
                lines.append(f"  OutReg = {I['srli']}(InReg, {remaining});")
            else:
                lines.append("  OutReg = InReg;")
            bit_pos = bit - remaining

        if bit_pos == LANE_BITS:
            lines.append(f"  {I['store']}(out++, OutReg);")
            bit_pos = 0

    if bit_pos > 0:
        lines.append(f"  {I['store']}(out++, OutReg);")

    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def gen_unpack_function(bit, I, mode='plain'):
    """Generate fused unpack+sum function for given bit width.

    mode:
      'plain'                 — aggregate_sums_u16(OutReg, sum)
      'corrected'             — aggregate_sums_u16_corrected(OutReg, correction, sum)
      'corrected_uniform'     — like corrected but takes a single broadcast anchor
                                instead of a per-OutReg corrections array. Used by
                                FoR-global (all OutRegs share the same anchor).
      'corrected_delta_local' — correction → zigzag_dec → per-OutReg prefix sum
                                → aggregate. Used by delta-local codec.
      'corrected_delta_carry' — correction → zigzag_dec → prefix_sum → +carry
                                → update carry → aggregate. Used by delta-carry
                                codec.
    """
    assert mode in ('plain', 'corrected', 'corrected_uniform',
                    'corrected_delta_local', 'corrected_delta_carry')

    lines = []
    lanes = I['lanes']
    block = I['block']
    total_values = block // lanes

    if bit == 0:
        return ""

    suffix = {
        'plain': '',
        'corrected': '_corrected',
        'corrected_uniform': '_corrected_uniform',
        'corrected_delta_local': '_corrected_delta_local',
        'corrected_delta_carry': '_corrected_delta_carry',
    }[mode]
    needs_corrections = mode in ('corrected', 'corrected_delta_local',
                                  'corrected_delta_carry')
    needs_anchor = mode == 'corrected_uniform'
    needs_carry = mode == 'corrected_delta_carry'

    extra_param = ''
    if needs_corrections:
        extra_param += f", const {I['reg']} *__restrict__ corrections"
    if needs_anchor:
        extra_param += f", {I['reg']} anchor"
    if needs_carry:
        extra_param += f", {I['reg']} *__restrict__ carry"

    def agg_call(reg_name, outreg_idx):
        if mode == 'plain':
            return f"  aggregate_sums_u16({reg_name}, sum);"
        if mode == 'corrected':
            return (f"  aggregate_sums_u16_corrected({reg_name}, "
                    f"corrections[{outreg_idx}], sum);")
        if mode == 'corrected_uniform':
            return f"  aggregate_sums_u16_corrected({reg_name}, anchor, sum);"
        if mode == 'corrected_delta_local':
            return (f"  agg_pipeline_local({reg_name}, "
                    f"corrections[{outreg_idx}], sum);")
        # corrected_delta_carry
        return (f"  agg_pipeline_carry({reg_name}, "
                f"corrections[{outreg_idx}], carry, sum);")

    def agg_call_for_outreg(outreg_idx):
        return agg_call("OutReg", outreg_idx)

    def agg_call_inreg(outreg_idx):
        # For bit == LANE_BITS the loaded InReg is conceptually the OutReg
        # (raw copy). For plain/corrected we can pass InReg directly. For
        # delta modes the pipeline mutates OutReg internally, so we still
        # pass it through the same helper.
        return agg_call("InReg", outreg_idx)

    lines.append(f"static void __SIMD_fastunpack{bit}_16{suffix}("
                 f"const {I['reg']} *__restrict__ in,")
    lines.append(f"    uint16_t *__restrict__ _out{extra_param}, "
                 f"{I['reg']} *__restrict__ sum) {{")
    lines.append("  (void)_out;")
    lines.append(f"  {I['reg']} InReg = {I['load']}(in);")

    if bit == LANE_BITS:
        lines.append(agg_call_inreg(0))
        for i in range(1, total_values):
            lines.append(f"  InReg = {I['load']}(++in);")
            lines.append(agg_call_inreg(i))
        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"  {I['reg']} OutReg;")
    lines.append(f"  const {I['reg']} mask = {I['set1']}((1U << {bit}) - 1);")
    lines.append("")

    bit_pos = 0

    for v in range(total_values):
        remaining = LANE_BITS - bit_pos

        if remaining >= bit:
            if bit_pos == 0:
                lines.append(f"  OutReg = {I['and_fn']}(InReg, mask);")
            else:
                lines.append(f"  OutReg = {I['and_fn']}({I['srli']}(InReg, {bit_pos}), mask);")
            lines.append(agg_call_for_outreg(v))
            lines.append("")
            bit_pos += bit
        else:
            if remaining > 0:
                lines.append(f"  OutReg = {I['srli']}(InReg, {bit_pos});")
            lines.append(f"  InReg = {I['load']}(++in);")
            need = bit - remaining
            if remaining > 0:
                lines.append(f"  OutReg =")
                lines.append(f"      {I['or_fn']}(OutReg, {I['and_fn']}({I['slli']}(InReg, {bit} - {need}), mask));")
            else:
                lines.append(f"  OutReg = {I['and_fn']}(InReg, mask);")
            lines.append(agg_call_for_outreg(v))
            lines.append("")
            bit_pos = need

        if bit_pos == LANE_BITS:
            if v < total_values - 1:
                lines.append(f"  InReg = {I['load']}(++in);")
            bit_pos = 0

    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def gen_dispatchers(I):
    lines = []

    # Pack dispatcher
    lines.append("} // namespace simdunaligned_u16")
    lines.append("")
    lines.append(f"void usimdpack_u16(const uint16_t *__restrict__ in, {I['reg']} *__restrict__ out,")
    lines.append(f"                   const uint32_t bit) {{")
    lines.append("  using namespace simdunaligned_u16;")
    lines.append("  switch (bit) {")
    for b in range(MAX_BIT + 1):
        lines.append(f"  case {b}:")
        lines.append(f"    __SIMD_fastpackwithoutmask{b}_16(in, out);")
        lines.append("    return;")
    lines.append("  default:")
    lines.append("    break;")
    lines.append("  }")
    lines.append("}")
    lines.append("")

    # Unpack dispatcher
    lines.append(f"void usimdunpack_u16(const {I['reg']} *__restrict__ in, uint16_t *__restrict__ out,")
    lines.append(f"                     const uint32_t bit, {I['reg']} *__restrict__ sum) {{")
    lines.append("  using namespace simdunaligned_u16;")
    lines.append("  switch (bit) {")
    lines.append("  case 0:")
    lines.append("    SIMD_nullunpacker16(in, out);")
    lines.append("    return;")
    for b in range(1, MAX_BIT + 1):
        lines.append(f"  case {b}:")
        lines.append(f"    __SIMD_fastunpack{b}_16(in, out, sum);")
        lines.append("    return;")
    lines.append("  default:")
    lines.append("    break;")
    lines.append("  }")
    lines.append("}")
    lines.append("")

    # Corrected unpack dispatcher (per-OutReg corrections folded into aggregation)
    outregs_per_block = I['block'] // I['lanes']
    lines.append(f"void usimdunpack_u16_corrected(const {I['reg']} *__restrict__ in,")
    lines.append(f"                                uint16_t *__restrict__ out,")
    lines.append(f"                                const uint32_t bit,")
    lines.append(f"                                const {I['reg']} *__restrict__ corrections,")
    lines.append(f"                                {I['reg']} *__restrict__ sum) {{")
    lines.append("  using namespace simdunaligned_u16;")
    lines.append("  (void)out;")
    lines.append("  switch (bit) {")
    lines.append("  case 0:")
    lines.append("    // b==0: every packed value is 0, every real value lives in `corrections`.")
    lines.append("    // corrections[lane] = exc (since exc - 0 = exc) at exception lanes, 0 elsewhere.")
    lines.append(f"    for (size_t i = 0; i < {outregs_per_block}; ++i) {{")
    lines.append("      aggregate_sums_u16(corrections[i], sum);")
    lines.append("    }")
    lines.append("    (void)in;")
    lines.append("    return;")
    for b in range(1, MAX_BIT + 1):
        lines.append(f"  case {b}:")
        lines.append(f"    __SIMD_fastunpack{b}_16_corrected(in, out, corrections, sum);")
        lines.append("    return;")
    lines.append("  default:")
    lines.append("    break;")
    lines.append("  }")
    lines.append("}")
    lines.append("")

    # corrected_uniform dispatcher: single broadcast anchor for all OutRegs.
    # Used by FoR-global exception-free path — no corrections array needed.
    lines.append(f"void usimdunpack_u16_corrected_uniform(const {I['reg']} *__restrict__ in,")
    lines.append(f"                                        uint16_t *__restrict__ out,")
    lines.append(f"                                        const uint32_t bit,")
    lines.append(f"                                        {I['reg']} anchor,")
    lines.append(f"                                        {I['reg']} *__restrict__ sum) {{")
    lines.append("  using namespace simdunaligned_u16;")
    lines.append("  (void)out;")
    lines.append("  switch (bit) {")
    lines.append("  case 0:")
    lines.append("    // b==0: all residuals are 0; anchor is the FoR base for every element.")
    lines.append(f"    for (size_t i = 0; i < {outregs_per_block}; ++i) {{")
    lines.append("      aggregate_sums_u16(anchor, sum);")
    lines.append("    }")
    lines.append("    (void)in;")
    lines.append("    return;")
    for b in range(1, MAX_BIT + 1):
        lines.append(f"  case {b}:")
        lines.append(f"    __SIMD_fastunpack{b}_16_corrected_uniform(in, out, anchor, sum);")
        lines.append("    return;")
    lines.append("  default:")
    lines.append("    break;")
    lines.append("  }")
    lines.append("}")
    lines.append("")

    # AVX2-only: delta-local / delta-carry dispatchers.
    if I['width'] == 256:
        # corrected_delta_local dispatcher
        lines.append(f"void usimdunpack_u16_corrected_delta_local("
                     f"const {I['reg']} *__restrict__ in,")
        lines.append(f"                                            uint16_t *__restrict__ out,")
        lines.append(f"                                            const uint32_t bit,")
        lines.append(f"                                            const {I['reg']} *__restrict__ corrections,")
        lines.append(f"                                            {I['reg']} *__restrict__ sum) {{")
        lines.append("  using namespace simdunaligned_u16;")
        lines.append("  (void)out;")
        lines.append("  switch (bit) {")
        lines.append("  case 0:")
        lines.append("    // b==0: unpacked OutReg is zero; corrections holds zigzag-deltas.")
        lines.append(f"    for (size_t i = 0; i < {outregs_per_block}; ++i) {{")
        lines.append("      agg_pipeline_local(_mm256_setzero_si256(), corrections[i], sum);")
        lines.append("    }")
        lines.append("    (void)in;")
        lines.append("    return;")
        for b in range(1, MAX_BIT + 1):
            lines.append(f"  case {b}:")
            lines.append(f"    __SIMD_fastunpack{b}_16_corrected_delta_local(in, out, corrections, sum);")
            lines.append("    return;")
        lines.append("  default:")
        lines.append("    break;")
        lines.append("  }")
        lines.append("}")
        lines.append("")

        # corrected_delta_carry dispatcher
        lines.append(f"void usimdunpack_u16_corrected_delta_carry("
                     f"const {I['reg']} *__restrict__ in,")
        lines.append(f"                                            uint16_t *__restrict__ out,")
        lines.append(f"                                            const uint32_t bit,")
        lines.append(f"                                            const {I['reg']} *__restrict__ corrections,")
        lines.append(f"                                            {I['reg']} *__restrict__ carry,")
        lines.append(f"                                            {I['reg']} *__restrict__ sum) {{")
        lines.append("  using namespace simdunaligned_u16;")
        lines.append("  (void)out;")
        lines.append("  switch (bit) {")
        lines.append("  case 0:")
        lines.append("    // b==0: unpacked OutReg is zero; corrections holds zigzag-deltas.")
        lines.append(f"    for (size_t i = 0; i < {outregs_per_block}; ++i) {{")
        lines.append("      agg_pipeline_carry(_mm256_setzero_si256(), corrections[i], carry, sum);")
        lines.append("    }")
        lines.append("    (void)in;")
        lines.append("    return;")
        for b in range(1, MAX_BIT + 1):
            lines.append(f"  case {b}:")
            lines.append(f"    __SIMD_fastunpack{b}_16_corrected_delta_carry(in, out, corrections, carry, sum);")
            lines.append("    return;")
        lines.append("  default:")
        lines.append("    break;")
        lines.append("  }")
        lines.append("}")
        lines.append("")

    lines.append("} // namespace FastPForLib")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Generate SIMD unaligned bit-packing for uint16')
    parser.add_argument('--width', type=int, choices=[128, 256], default=128,
                        help='Register width in bits (128=AVX, 256=AVX2)')
    args = parser.parse_args()

    I = intrinsics(args.width)
    lanes = args.width // LANE_BITS
    block = lanes * MAX_BIT
    I['lanes'] = lanes
    I['block'] = block
    I['width'] = args.width

    out = []
    out.append(gen_header(I))

    for bit in range(MAX_BIT + 1):
        out.append(gen_pack_function(bit, I))

    for bit in range(1, MAX_BIT + 1):
        out.append(gen_unpack_function(bit, I, mode='plain'))

    for bit in range(1, MAX_BIT + 1):
        out.append(gen_unpack_function(bit, I, mode='corrected'))

    for bit in range(1, MAX_BIT + 1):
        out.append(gen_unpack_function(bit, I, mode='corrected_uniform'))

    if I['width'] == 256:
        for bit in range(1, MAX_BIT + 1):
            out.append(gen_unpack_function(bit, I, mode='corrected_delta_local'))

        for bit in range(1, MAX_BIT + 1):
            out.append(gen_unpack_function(bit, I, mode='corrected_delta_carry'))

    out.append(gen_dispatchers(I))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(script_dir, '..', 'src')
    out_path = os.path.join(src_dir, 'simdunalignedbitpacking_u16.cpp')

    with open(out_path, 'w') as f:
        f.write("".join(out))
    print(f"Generated {out_path}")


if __name__ == "__main__":
    main()
