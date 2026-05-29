#!/usr/bin/env python3
"""
Generate SIMD unaligned bit-packing for uint32 with configurable register width.

Generates two files:
  1. simdunalignedbitpacking.cpp — pack (with/without mask) + unpack (stores output)
  2. simdunalignedbitpacking_new.cpp — fused unpack with sum aggregation (no output stores)

Usage:
  python gen_simdunalignedbitpacking.py --width 128   # AVX (128-bit registers)
  python gen_simdunalignedbitpacking.py --width 256   # AVX2 (256-bit registers)
"""
import argparse
import os
import sys

LANE_BITS = 32
MAX_BIT = 32


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
            slli='_mm_slli_epi32',
            srli='_mm_srli_epi32',
            set1='_mm_set1_epi32',
            add='_mm_add_epi32',
        )
    else:
        return dict(
            reg='__m256i',
            load='_mm256_loadu_si256',
            store='_mm256_storeu_si256',
            setzero='_mm256_setzero_si256',
            or_fn='_mm256_or_si256',
            and_fn='_mm256_and_si256',
            slli='_mm256_slli_epi32',
            srli='_mm256_srli_epi32',
            set1='_mm256_set1_epi32',
            add='_mm256_add_epi32',
        )


def gen_pack_withoutmask(bit, I):
    """Generate pack-without-mask function for given bit width."""
    lines = []
    lanes = I['lanes']
    block = I['block']
    total_inputs = block // lanes  # always 32

    if bit == 0:
        lines.append(f"static void __SIMD_fastpackwithoutmask0_32("
                     f"const uint32_t *__restrict__, {I['reg']} *__restrict__) {{")
        lines.append("}")
        return "\n".join(lines)

    lines.append(f"static void __SIMD_fastpackwithoutmask{bit}_32("
                 f"const uint32_t *__restrict__ _in,")
    lines.append(f"    {I['reg']} *__restrict__ out) {{")
    lines.append(f"  const {I['reg']} *in = reinterpret_cast<const {I['reg']} *>(_in);")

    if bit == LANE_BITS:
        lines.append(f"  for (int i = 0; i < {total_inputs}; ++i) {{")
        lines.append(f"    {I['store']}(out + i, {I['load']}(in + i));")
        lines.append("  }")
        lines.append("}")
        return "\n".join(lines)

    lines.append(f"  {I['reg']} OutReg;")
    lines.append(f"  {I['reg']} InReg = {I['load']}(in);")
    lines.append("")
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
    return "\n".join(lines)


def gen_pack_withmask(bit, I):
    """Generate pack-with-mask function for given bit width."""
    lines = []
    lanes = I['lanes']
    block = I['block']
    total_inputs = block // lanes  # always 32

    if bit == 0:
        return ""  # no 0-bit pack with mask

    lines.append(f"static void __SIMD_fastpack{bit}_32("
                 f"const uint32_t *__restrict__ _in,")
    lines.append(f"    {I['reg']} *__restrict__ out) {{")
    lines.append(f"  const {I['reg']} *in = reinterpret_cast<const {I['reg']} *>(_in);")
    lines.append(f"  {I['reg']} OutReg;")
    lines.append("")

    if bit == LANE_BITS:
        lines.append(f"  for (int i = 0; i < {total_inputs}; ++i) {{")
        lines.append(f"    {I['store']}(out + i, {I['load']}(in + i));")
        lines.append("  }")
        lines.append("}")
        return "\n".join(lines)

    lines.append(f"  const {I['reg']} mask = {I['set1']}((1U << {bit}) - 1);")
    lines.append("")
    lines.append(f"  {I['reg']} InReg = {I['and_fn']}({I['load']}(in), mask);")
    lines.append("  OutReg = InReg;")

    bit_pos = bit
    for i in range(1, total_inputs):
        lines.append(f"  InReg = {I['and_fn']}({I['load']}(++in), mask);")
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
            lines.append(f"  {I['store']}(out, OutReg);")
            lines.append("  ++out;")
            if remaining > 0:
                lines.append(f"  OutReg = {I['srli']}(InReg, {remaining});")
            else:
                lines.append("  OutReg = InReg;")
            bit_pos = bit - remaining

        if bit_pos == LANE_BITS:
            lines.append(f"  {I['store']}(out, OutReg);")
            lines.append("  ++out;")
            bit_pos = 0

    if bit_pos > 0:
        lines.append(f"  {I['store']}(out, OutReg);")
        lines.append("  ++out;")

    lines.append("}")
    return "\n".join(lines)


def gen_unpack(bit, I):
    """Generate unpack function that stores to output."""
    lines = []
    lanes = I['lanes']
    block = I['block']
    total_outputs = block // lanes  # always 32

    if bit == 0:
        return ""  # handled by nullunpacker

    lines.append(f"static void __SIMD_fastunpack{bit}_32("
                 f"const {I['reg']} *__restrict__ in,")
    lines.append(f"    uint32_t *__restrict__ _out) {{")
    lines.append(f"  {I['reg']} *out = reinterpret_cast<{I['reg']} *>(_out);")

    if bit == LANE_BITS:
        lines.append(f"  for (uint32_t outer = 0; outer < {total_outputs}; ++outer) {{")
        lines.append(f"    {I['store']}(out++, {I['load']}(in++));")
        lines.append("  }")
        lines.append("}")
        return "\n".join(lines)

    lines.append(f"  {I['reg']} InReg = {I['load']}(in);")
    lines.append(f"  {I['reg']} OutReg;")
    lines.append(f"  const {I['reg']} mask = {I['set1']}((1U << {bit}) - 1);")
    lines.append("")

    bit_pos = 0
    for v in range(total_outputs):
        remaining = LANE_BITS - bit_pos

        if remaining >= bit:
            if bit_pos == 0:
                lines.append(f"  OutReg = {I['and_fn']}(InReg, mask);")
            else:
                lines.append(f"  OutReg = {I['and_fn']}({I['srli']}(InReg, {bit_pos}), mask);")
            lines.append(f"  {I['store']}(out++, OutReg);")
            lines.append("")
            bit_pos += bit
        else:
            # Spans two input registers
            if remaining > 0:
                lines.append(f"  OutReg = {I['srli']}(InReg, {bit_pos});")
            lines.append(f"  InReg = {I['load']}(++in);")
            need = bit - remaining
            if remaining > 0:
                lines.append(f"  OutReg =")
                lines.append(f"      {I['or_fn']}(OutReg, {I['and_fn']}({I['slli']}(InReg, {bit} - {need}), mask));")
            else:
                lines.append(f"  OutReg = {I['and_fn']}(InReg, mask);")
            lines.append(f"  {I['store']}(out++, OutReg);")
            lines.append("")
            bit_pos = need

        if bit_pos == LANE_BITS:
            if v < total_outputs - 1:
                lines.append(f"  InReg = {I['load']}(++in);")
            bit_pos = 0

    lines.append("}")
    return "\n".join(lines)


def gen_unpack_sum(bit, I):
    """Generate fused unpack+sum function (no output stores)."""
    lines = []
    lanes = I['lanes']
    block = I['block']
    total_outputs = block // lanes  # always 32

    if bit == 0:
        return ""  # handled by nullunpacker

    lines.append(f"static void __SIMD_fastunpack{bit}_32("
                 f"const {I['reg']} *in, uint32_t *_out, {I['reg']}* sum) {{")
    lines.append(f"  (void)_out;")

    if bit == LANE_BITS:
        lines.append(f"  for (uint32_t outer = 0; outer < {total_outputs}; ++outer) {{")
        lines.append(f"    aggregate_sums({I['load']}(in++), sum);")
        lines.append("  }")
        lines.append("}")
        return "\n".join(lines)

    lines.append(f"  {I['reg']} InReg = {I['load']}(in);")
    lines.append(f"  {I['reg']} OutReg;")
    lines.append(f"  const {I['reg']} mask = {I['set1']}((1U << {bit}) - 1);")
    lines.append("")

    bit_pos = 0
    for v in range(total_outputs):
        remaining = LANE_BITS - bit_pos

        if remaining >= bit:
            if bit_pos == 0:
                lines.append(f"  OutReg = {I['and_fn']}(InReg, mask);")
            else:
                lines.append(f"  OutReg = {I['and_fn']}({I['srli']}(InReg, {bit_pos}), mask);")
            lines.append("  aggregate_sums(OutReg, sum);")
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
            lines.append("  aggregate_sums(OutReg, sum);")
            lines.append("")
            bit_pos = need

        if bit_pos == LANE_BITS:
            if v < total_outputs - 1:
                lines.append(f"  InReg = {I['load']}(++in);")
            bit_pos = 0

    lines.append("}")
    return "\n".join(lines)


def gen_file_main(I):
    """Generate simdunalignedbitpacking.cpp content."""
    out = []
    out.append(f"""\
/**
 * Auto-generated uint32 SIMD unaligned bit-packing.
 * Generated by gen_simdunalignedbitpacking.py (width={I['width']})
 *
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 */
#include "usimdbitpacking.h"

namespace FastPForLib {{

namespace simdunaligned {{

static void SIMD_nullunpacker32(const {I['reg']} *__restrict__,
                                uint32_t *__restrict__ out) {{
  memset(out, 0, {I['block']} * 4);
}}

""")

    # Pack without mask: 0..32
    for bit in range(MAX_BIT + 1):
        out.append(gen_pack_withoutmask(bit, I))
        out.append("")

    # Pack with mask: 1..32
    for bit in range(1, MAX_BIT + 1):
        out.append(gen_pack_withmask(bit, I))
        out.append("")

    # Unpack: 1..32
    for bit in range(1, MAX_BIT + 1):
        out.append(gen_unpack(bit, I))
        out.append("")

    # Dispatchers
    out.append("} // namespace simdunaligned")
    out.append("")

    # usimdunpack dispatcher
    out.append(f"void usimdunpack(const {I['reg']} *__restrict__ in, uint32_t *__restrict__ out,")
    out.append(f"                 const uint32_t bit) {{")
    out.append("  using namespace simdunaligned;")
    out.append("  switch (bit) {")
    out.append("  case 0:")
    out.append("    SIMD_nullunpacker32(in, out);")
    out.append("    return;")
    for b in range(1, MAX_BIT + 1):
        out.append(f"  case {b}:")
        out.append(f"    __SIMD_fastunpack{b}_32(in, out);")
        out.append("    return;")
    out.append("  default:")
    out.append("    break;")
    out.append("  }")
    out.append('  throw std::logic_error("number of bits is unsupported");')
    out.append("}")
    out.append("")

    # usimdpackwithoutmask dispatcher
    out.append(f"void usimdpackwithoutmask(const uint32_t *__restrict__ in,")
    out.append(f"                          {I['reg']} *__restrict__ out, const uint32_t bit) {{")
    out.append("  using namespace simdunaligned;")
    out.append("  switch (bit) {")
    out.append("  case 0:")
    out.append("    return;")
    for b in range(1, MAX_BIT + 1):
        out.append(f"  case {b}:")
        out.append(f"    __SIMD_fastpackwithoutmask{b}_32(in, out);")
        out.append("    return;")
    out.append("  default:")
    out.append("    break;")
    out.append("  }")
    out.append('  throw std::logic_error("number of bits is unsupported");')
    out.append("}")
    out.append("")

    # usimdpack dispatcher
    out.append(f"void usimdpack(const uint32_t *__restrict__ in, {I['reg']} *__restrict__ out,")
    out.append(f"               const uint32_t bit) {{")
    out.append("  using namespace simdunaligned;")
    out.append("  switch (bit) {")
    out.append("  case 0:")
    out.append("    return;")
    for b in range(1, MAX_BIT + 1):
        out.append(f"  case {b}:")
        out.append(f"    __SIMD_fastpack{b}_32(in, out);")
        out.append("    return;")
    out.append("  default:")
    out.append("    break;")
    out.append("  }")
    out.append('  throw std::logic_error("number of bits is unsupported");')
    out.append("}")
    out.append("")
    out.append("} // namespace FastPForLib")
    out.append("")

    return "\n".join(out)


def gen_file_new(I):
    """Generate simdunalignedbitpacking_new.cpp content."""
    out = []
    out.append(f"""\
/**
 * Auto-generated uint32 SIMD unaligned bit-packing with fused sum aggregation.
 * Generated by gen_simdunalignedbitpacking.py (width={I['width']})
 *
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 */
#include "usimdbitpacking_new.h"

namespace FastPForLib {{

namespace simdunaligned_new {{

static void SIMD_nullunpacker32(const {I['reg']} *__restrict__,
                                uint32_t *__restrict__ out) {{
}}

static inline void aggregate_sums({I['reg']} OutReg, {I['reg']}* sum) {{
    *sum = {I['add']}(*sum, OutReg);
}}

""")

    # Unpack with sum: 1..32
    for bit in range(1, MAX_BIT + 1):
        out.append(gen_unpack_sum(bit, I))
        out.append("")

    # Dispatcher
    out.append("} // namespace simdunaligned_new")
    out.append("")
    out.append(f"void usimdunpack_new(const {I['reg']} *__restrict__ in, uint32_t *__restrict__ out,")
    out.append(f"                 const uint32_t bit, {I['reg']}* sum) {{")
    out.append("  using namespace simdunaligned_new;")
    out.append("  switch (bit) {")
    out.append("  case 0:")
    out.append("    SIMD_nullunpacker32(in, out);")
    out.append("    return;")
    for b in range(1, MAX_BIT + 1):
        out.append(f"  case {b}:")
        out.append(f"    __SIMD_fastunpack{b}_32(in, out, sum);")
        out.append("    return;")
    out.append("  default:")
    out.append("    break;")
    out.append("  }")
    out.append('  throw std::logic_error("number of bits is unsupported");')
    out.append("}")
    out.append("")
    out.append("} // namespace FastPForLib")
    out.append("")

    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(
        description='Generate SIMD unaligned bit-packing for uint32')
    parser.add_argument('--width', type=int, choices=[128, 256], required=True,
                        help='Register width in bits (128=AVX, 256=AVX2)')
    args = parser.parse_args()

    I = intrinsics(args.width)
    lanes = args.width // LANE_BITS
    block = lanes * MAX_BIT
    I['lanes'] = lanes
    I['block'] = block
    I['width'] = args.width

    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(script_dir, '..', 'src')

    main_path = os.path.join(src_dir, 'simdunalignedbitpacking.cpp')
    new_path = os.path.join(src_dir, 'simdunalignedbitpacking_new.cpp')

    with open(main_path, 'w') as f:
        f.write(gen_file_main(I))
    print(f"Generated {main_path}")

    with open(new_path, 'w') as f:
        f.write(gen_file_new(I))
    print(f"Generated {new_path}")


if __name__ == "__main__":
    main()
