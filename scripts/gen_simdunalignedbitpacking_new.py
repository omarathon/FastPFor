#!/usr/bin/env python3
"""
Transform simdunalignedbitpacking_new.cpp to add compile-time SIMD_SUM_FUSED
mode toggle.

The source file was hand-edited to comment out all unpack output stores and add
aggregate_sums() calls.  This script adds #ifdef guards so the file can be
compiled in either mode:

  #define SIMD_SUM_FUSED   → sum-only (no output writes, original behaviour)
  (undefined)              → normal decode (output written, no sum)

NOTE: Pack functions are untouched — their `/* _mm_storeu_si128(out, OutReg); */
      ++out;` pattern does NOT have an adjacent aggregate_sums call, so it is
      left as-is in all modes.

Usage:
  python3 gen_simdunalignedbitpacking_new.py [--mode {fused,normal,guarded}] \
      [--input simdunalignedbitpacking_new.cpp] \
      [--output simdunalignedbitpacking_new.cpp]

  fused   — emit the current sum-only code (no #ifdef, hardcoded fused)
  normal  — emit normal decode code (no #ifdef, hardcoded normal)
  guarded — (default) emit #ifdef SIMD_SUM_FUSED guarded code
"""

import argparse
import re
import sys


# ---------------------------------------------------------------------------
# Patterns
#
# Unpack stores come in two comment styles, always followed by aggregate_sums:
#
# Style A (loop form, double-slash, trailing space):
#   <indent>//  _mm_storeu_si128(out++, <name>); \n
#   <indent>aggregate_sums(<name>, sum);
#
# Style B (/* */ form, variable indent before aggregate_sums):
#   <indent>/* _mm_storeu_si128(out++, <name>); */\n
#   <indent2>aggregate_sums(<name>, sum);
#
# Both require the NEXT line to be aggregate_sums to distinguish from pack
# function stores (which are followed by ++out or nothing).
# ---------------------------------------------------------------------------

# Matches Style A: //  _mm_storeu_si128(out++, <reg>); \n<indent>aggregate_sums(<reg>, sum);
STYLE_A_RE = re.compile(
    r'(?P<indent>[ \t]*)//  _mm_storeu_si128\(out\+\+, (?P<reg>\w+)\); \n'
    r'(?P<indent2>[ \t]*)aggregate_sums\((?P=reg), sum\);',
    re.MULTILINE
)

# Matches Style B: /* _mm_storeu_si128(out++, <reg>); */\n<indent2>aggregate_sums(<reg>, sum);
STYLE_B_RE = re.compile(
    r'(?P<indent>[ \t]*)/\* _mm_storeu_si128\(out\+\+, (?P<reg>\w+)\); \*/\n'
    r'(?P<indent2>[ \t]*)aggregate_sums\((?P=reg), sum\);',
    re.MULTILINE
)

# Matches the `out` declaration in unpack functions
OUT_DECL_RE = re.compile(
    r'([ \t]*)__m128i \*out = reinterpret_cast<__m128i \*>\(_out\);'
)


def make_replacement(mode):
    def replacement(m):
        ind = m.group('indent')
        ind2 = m.group('indent2')
        reg = m.group('reg')
        if mode == 'guarded':
            return (
                f'{ind}#ifdef SIMD_SUM_FUSED\n'
                f'{ind2}aggregate_sums({reg}, sum);\n'
                f'{ind}#else\n'
                f'{ind}_mm_storeu_si128(out++, {reg});\n'
                f'{ind}#endif'
            )
        elif mode == 'normal':
            return f'{ind}_mm_storeu_si128(out++, {reg});'
        else:  # fused
            return f'{ind2}aggregate_sums({reg}, sum);'
    return replacement


def out_decl_replacement(mode):
    def replacement(m):
        ind = m.group(1)
        if mode == 'guarded':
            return (
                f'{ind}#ifdef SIMD_SUM_FUSED\n'
                f'{ind}(void)_out;\n'
                f'{ind}#else\n'
                f'{ind}__m128i *out = reinterpret_cast<__m128i *>(_out);\n'
                f'{ind}#endif'
            )
        elif mode == 'normal':
            return f'{ind}__m128i *out = reinterpret_cast<__m128i *>(_out);'
        else:  # fused — remove, replace with (void)_out
            return f'{ind}(void)_out;'
    return replacement


def transform(src, mode):
    repl = make_replacement(mode)
    src = STYLE_A_RE.sub(repl, src)
    src = STYLE_B_RE.sub(repl, src)
    src = OUT_DECL_RE.sub(out_decl_replacement(mode), src)
    return src


def main():
    parser = argparse.ArgumentParser(
        description='Transform simdunalignedbitpacking_new.cpp')
    parser.add_argument('--mode', choices=['fused', 'normal', 'guarded'],
                        default='guarded')
    parser.add_argument('--input', default=None,
                        help='Input file (default: stdin)')
    parser.add_argument('--output', default=None,
                        help='Output file (default: stdout)')
    args = parser.parse_args()

    src = open(args.input).read() if args.input else sys.stdin.read()
    result = transform(src, args.mode)
    if args.output:
        open(args.output, 'w').write(result)
    else:
        sys.stdout.write(result)


if __name__ == '__main__':
    main()
