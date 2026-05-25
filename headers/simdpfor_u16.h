/**
 * uint16 variant of SIMDPFor with fused sum aggregation.
 * Adapted from simdpfor.h for 16-bit data.
 *
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 */

#ifndef SIMDPFOR_U16_H_
#define SIMDPFOR_U16_H_

#include "common.h"
#include "usimdbitpacking_u16.h"
#include "util.h"
#include <cstdlib>
#include <vector>

namespace FastPForLib {

class SIMDPForU16 {
public:
  enum {
    BlockSizeInUnitsOfPackSize = 16,
    PACKSIZE = 16,
    BlockSize = BlockSizeInUnitsOfPackSize * PACKSIZE, // 256
    blocksizeinbits = 8
  };

  typedef uint16_t DATATYPE;

  std::vector<uint16_t> codedcopy;
  std::vector<uint32_t> miss;

  SIMDPForU16() : codedcopy(BlockSize), miss(BlockSize) {}

  static uint32_t determineBestBase(const DATATYPE *in, size_t size) {
    if (size == 0)
      return 0;
    const size_t defaultsamplesize = 64 * 1024;
    size_t samplesize = size > defaultsamplesize ? defaultsamplesize : size;
    uint32_t freqs[17];
    for (uint32_t k = 0; k <= 16; ++k)
      freqs[k] = 0;
    uint32_t rstart =
        size > samplesize
            ? (rand() % (static_cast<uint32_t>(size - samplesize)))
            : 0U;
    for (uint32_t k = rstart; k < rstart + samplesize; ++k) {
      freqs[bits(static_cast<uint32_t>(in[k]))]++;
    }
    uint32_t bestb = 16;
    uint32_t numberofexceptions = 0;
    double Erate = 0;
    double bestcost = 16;
    for (uint32_t b = bestb - 1; b < 16; --b) {
      numberofexceptions += freqs[b + 1];
      Erate = static_cast<double>(numberofexceptions) /
              static_cast<double>(samplesize);
      if (numberofexceptions > 0) {
        double altErate = (Erate * 128 - 1) / (Erate * (1U << b));
        if (altErate > Erate)
          Erate = altErate;
      }
      const double thiscost = b + Erate * 16;
      if (thiscost <= bestcost) {
        bestcost = thiscost;
        bestb = b;
      }
    }
    return bestb;
  }

  uint32_t compressblockPFOR(const DATATYPE *__restrict__ in,
                             uint32_t *__restrict__ outputbegin,
                             const uint32_t b,
                             DATATYPE *__restrict__ &exceptions) {
    if (b == 16) {
      // raw copy: store 128 uint16 values = 64 uint32 words
      const uint16_t *src = in;
      for (size_t k = 0; k < BlockSize / 2; ++k) {
        outputbegin[k] = static_cast<uint32_t>(src[2 * k]) |
                         (static_cast<uint32_t>(src[2 * k + 1]) << 16);
      }
      return BlockSize;
    }
    size_t exceptcounter = 0;
    const uint32_t maxgap = 1U << b;
    {
      for (uint32_t k = 0; k < BlockSize; ++k) {
        miss[exceptcounter] = k;
        exceptcounter += (in[k] >= maxgap);
      }
    }
    if (exceptcounter == 0) {
      packblock(in, outputbegin, b);
      return BlockSize;
    }
    codedcopy.assign(in, in + BlockSize);
    uint32_t firstexcept = miss[0];
    uint32_t prev = 0;
    *(exceptions++) = codedcopy[firstexcept];
    prev = firstexcept;
    if (maxgap < BlockSize) {
      for (uint32_t i = 1; i < exceptcounter; ++i) {
        uint32_t cur = miss[i];
        while (cur > maxgap + prev) {
          uint32_t compulcur = prev + maxgap;
          *(exceptions++) = codedcopy[compulcur];
          codedcopy[prev] = static_cast<uint16_t>(maxgap - 1);
          prev = compulcur;
        }
        *(exceptions++) = codedcopy[cur];
        codedcopy[prev] = static_cast<uint16_t>(cur - prev - 1);
        prev = cur;
      }
    } else {
      for (uint32_t i = 1; i < exceptcounter; ++i) {
        uint32_t cur = miss[i];
        *(exceptions++) = codedcopy[cur];
        codedcopy[prev] = static_cast<uint16_t>(cur - prev - 1);
        prev = cur;
      }
    }
    codedcopy[prev] &= static_cast<uint16_t>((1U << b) - 1);
    packblock(&codedcopy[0], outputbegin, b);
    return firstexcept;
  }

  void packblock(const uint16_t *source, uint32_t *out, const uint32_t bit) {
    usimdpack_u16(source, reinterpret_cast<__m256i *>(out), bit);
  }

  void unpackblock(const uint32_t *source, uint16_t *out, const uint32_t bit,
                   __m256i *sum) {
    (void)out;
    usimdunpack_u16(reinterpret_cast<const __m256i *>(source), out, bit, sum);
  }

  void unpackblock_corrected(const uint32_t *source, uint16_t *out,
                             const uint32_t bit,
                             const __m256i *corrections, __m256i *sum) {
    (void)out;
    usimdunpack_u16_corrected(reinterpret_cast<const __m256i *>(source), out,
                              bit, corrections, sum);
  }

  void encodeArray(const uint16_t *in, const size_t len, uint32_t *out,
                   size_t &nvalue) {
    *out++ = static_cast<uint32_t>(len);
    const uint32_t maxsize = (1U << (32 - blocksizeinbits - 1));
    size_t totalnvalue(1);
    for (size_t j = 0; j < (len + maxsize - 1U) / maxsize; ++j) {
      size_t i = j << (32 - blocksizeinbits - 1);
      size_t l = maxsize;
      if (i + maxsize > len) {
        l = len - i;
      }
      size_t thisnvalue = nvalue - totalnvalue;
      __encodeArray(&in[i], l, out, thisnvalue);
      totalnvalue += thisnvalue;
      out += thisnvalue;
    }
    nvalue = totalnvalue;
  }

  const uint32_t *decodeArray(const uint32_t *in, const size_t len,
                              uint16_t *out, size_t &nvalue) {
    nvalue = *in++;
    if (nvalue == 0) {
      return in;
    }
    const uint32_t *const finalin = in + len;
    size_t totalnvalue(0);
    __m256i sum = _mm256_setzero_si256();
    int32_t delta_sum = 0;
    uint16_t *initout = out;
    while (totalnvalue < nvalue) {
      size_t thisnvalue = nvalue - totalnvalue;
      in = __decodeArray(in, finalin - in, out, thisnvalue, &sum, &delta_sum);
      out += thisnvalue;
      totalnvalue += thisnvalue;
    }
    nvalue = totalnvalue;

    // 256-bit horizontal sum: fold 256->128, then 128->scalar
    __m128i lo = _mm256_castsi256_si128(sum);
    __m128i hi = _mm256_extracti128_si256(sum, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    const auto out_sum =
        static_cast<uint32_t>(_mm_cvtsi128_si32(s) + delta_sum);

    // store sum as uint32 in 2 uint16 slots after decoded data
    initout[nvalue] = static_cast<uint16_t>(out_sum & 0xFFFF);
    initout[nvalue + 1] = static_cast<uint16_t>(out_sum >> 16);

    return in;
  }

  // Corrected variant of decodeArray: folds exception correction into the
  // SIMD aggregation via per-OutReg correction masks. Removes the post-loop
  // delta_sum step entirely.
  const uint32_t *decodeArrayCorrected(const uint32_t *in, const size_t len,
                                        uint16_t *out, size_t &nvalue) {
    nvalue = *in++;
    if (nvalue == 0) {
      return in;
    }
    const uint32_t *const finalin = in + len;
    size_t totalnvalue(0);
    __m256i sum = _mm256_setzero_si256();
    uint16_t *initout = out;
    while (totalnvalue < nvalue) {
      size_t thisnvalue = nvalue - totalnvalue;
      in = __decodeArrayCorrected(in, finalin - in, out, thisnvalue, &sum);
      out += thisnvalue;
      totalnvalue += thisnvalue;
    }
    nvalue = totalnvalue;

    // 256-bit horizontal sum: fold 256->128, then 128->scalar
    __m128i lo = _mm256_castsi256_si128(sum);
    __m128i hi = _mm256_extracti128_si256(sum, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    const auto out_sum = static_cast<uint32_t>(_mm_cvtsi128_si32(s));

    // store sum as uint32 in 2 uint16 slots after decoded data
    initout[nvalue] = static_cast<uint16_t>(out_sum & 0xFFFF);
    initout[nvalue + 1] = static_cast<uint16_t>(out_sum >> 16);

    return in;
  }

  void __encodeArray(const uint16_t *in, const size_t len, uint32_t *out,
                     size_t &nvalue) {
    checkifdivisibleby(len, BlockSize);
    const uint32_t *const initout(out);
    std::vector<DATATYPE> exceptions;
    exceptions.resize(len);
    DATATYPE *__restrict__ i = &exceptions[0];
    const uint32_t b = determineBestBase(in, len);
    *out++ = static_cast<uint32_t>(len);
    *out++ = b;
    for (size_t k = 0; k < len / BlockSize; ++k) {
      uint32_t *const headerout(out);
      ++out;
      uint32_t firstexcept = compressblockPFOR(in, out, b, i);
      out += (BlockSize * b) / 32;
      in += BlockSize;
      const uint32_t bitsforfirstexcept = blocksizeinbits;
      const uint32_t firstexceptmask = (1U << blocksizeinbits) - 1;
      const uint32_t exceptindex = static_cast<uint32_t>(i - &exceptions[0]);
      *headerout =
          (firstexcept & firstexceptmask) | (exceptindex << bitsforfirstexcept);
    }
    // store exceptions: one uint16 per uint32 word (same layout as 32-bit)
    const size_t howmanyexcept = i - &exceptions[0];
    for (uint32_t t = 0; t < howmanyexcept; ++t)
      *out++ = static_cast<uint32_t>(exceptions[t]);
    nvalue = out - initout;
  }

  const uint32_t *__decodeArray(const uint32_t *in, const size_t len,
                                uint16_t *out, size_t &nvalue, __m256i *sum,
                                int32_t *delta_sum) {
    (void)len;
    nvalue = *in++;
    checkifdivisibleby(nvalue, BlockSize);
    const uint32_t b = *in++;
    const uint32_t *__restrict__ except =
        in + nvalue * b / 32 + nvalue / BlockSize;

    const uint32_t bitsforfirstexcept = blocksizeinbits;
    const uint32_t firstexceptmask = (1U << blocksizeinbits) - 1;

    // Decode exceptions on the fly using packed uint16 pairs
    size_t except_offset = 0;

    for (size_t k = 0; k < nvalue / BlockSize; ++k) {
      const uint32_t *const headerin(in);
      ++in;
      const uint32_t firstexcept = *headerin & firstexceptmask;
      const uint32_t exceptindex = *headerin >> bitsforfirstexcept;
      const size_t end_except_idx = exceptindex;

      uncompressblockPFOR_u16(in, out, b, except, except_offset,
                              end_except_idx, firstexcept, sum, delta_sum);
      except_offset = end_except_idx;
      in += (BlockSize * b) / 32;
      out += BlockSize;
    }

    // return pointer past exceptions (1 exception per uint32 word)
    return except + except_offset;
  }

  // Corrected variant: same control flow as __decodeArray, but each block
  // uses uncompressblockPFOR_u16_corrected which folds exception correction
  // into the SIMD aggregation (no delta_sum needed).
  const uint32_t *__decodeArrayCorrected(const uint32_t *in, const size_t len,
                                          uint16_t *out, size_t &nvalue,
                                          __m256i *sum) {
    (void)len;
    nvalue = *in++;
    checkifdivisibleby(nvalue, BlockSize);
    const uint32_t b = *in++;
    const uint32_t *__restrict__ except =
        in + nvalue * b / 32 + nvalue / BlockSize;

    const uint32_t bitsforfirstexcept = blocksizeinbits;
    const uint32_t firstexceptmask = (1U << blocksizeinbits) - 1;

    size_t except_offset = 0;

    for (size_t k = 0; k < nvalue / BlockSize; ++k) {
      const uint32_t *const headerin(in);
      ++in;
      const uint32_t firstexcept = *headerin & firstexceptmask;
      const uint32_t exceptindex = *headerin >> bitsforfirstexcept;
      const size_t end_except_idx = exceptindex;

      uncompressblockPFOR_u16_corrected(in, out, b, except, except_offset,
                                         end_except_idx, firstexcept, sum);
      except_offset = end_except_idx;
      in += (BlockSize * b) / 32;
      out += BlockSize;
    }

    return except + except_offset;
  }

  void uncompressblockPFOR_u16(const uint32_t *__restrict__ inputbegin,
                               uint16_t *__restrict__ outputbegin,
                               const uint32_t b,
                               const uint32_t *__restrict__ except_base,
                               size_t start_except_idx, size_t end_except_idx,
                               size_t next_exception, __m256i *sum,
                               int32_t *delta_sum) {
    if (b == 16) {
      // raw data: unpack and aggregate via zero-extension
      const uint16_t *raw = reinterpret_cast<const uint16_t *>(inputbegin);
      __m256i zero = _mm256_setzero_si256();
      for (size_t i = 0; i < BlockSize; i += 16) {
        __m256i v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(raw + i));
        *sum = _mm256_add_epi32(*sum, _mm256_unpacklo_epi16(v, zero));
        *sum = _mm256_add_epi32(*sum, _mm256_unpackhi_epi16(v, zero));
      }
      return;
    }

    // fused unpack: aggregate sums, don't write output
    unpackblock(inputbegin, outputbegin, b, sum);

    // correct exceptions
    const uint16_t *packed_data =
        reinterpret_cast<const uint16_t *>(inputbegin);
    for (size_t idx = start_except_idx; idx != end_except_idx;) {
      const auto gap = read_gap_simd_layout_u16(packed_data, b, next_exception);
      next_exception = next_exception + static_cast<size_t>(gap) + 1;

      uint16_t exc_val = static_cast<uint16_t>(except_base[idx]);

      *delta_sum += (static_cast<int32_t>(exc_val) - static_cast<int32_t>(gap));
      idx++;
    }
  }

  // Corrected variant: pre-computes per-OutReg correction masks from the
  // exception chain, then runs the corrected SIMD unpack-and-aggregate path.
  // Exception correction is folded into the hot loop (one uint16 ADD per
  // OutReg) instead of a post-loop scalar delta.
  void uncompressblockPFOR_u16_corrected(
      const uint32_t *__restrict__ inputbegin,
      uint16_t *__restrict__ outputbegin, const uint32_t b,
      const uint32_t *__restrict__ except_base, size_t start_except_idx,
      size_t end_except_idx, size_t next_exception, __m256i *sum) {
    if (b == 16) {
      // raw data: no exceptions possible at b=16. Same hot loop as the
      // non-corrected variant.
      const uint16_t *raw = reinterpret_cast<const uint16_t *>(inputbegin);
      __m256i zero = _mm256_setzero_si256();
      for (size_t i = 0; i < BlockSize; i += 16) {
        __m256i v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(raw + i));
        *sum = _mm256_add_epi32(*sum, _mm256_unpacklo_epi16(v, zero));
        *sum = _mm256_add_epi32(*sum, _mm256_unpackhi_epi16(v, zero));
      }
      return;
    }

    if (start_except_idx == end_except_idx) {
      // Exception-free block: skip the corrections setup and the per-OutReg
      // add — fall back to the standard fused unpack (matches the encoder's
      // exception-free fast path).
      unpackblock(inputbegin, outputbegin, b, sum);
      return;
    }

    // Stack-local correction array. Layout: one uint16 per element of the
    // block; nonzero only at exception positions. Read as 16 __m256i by the
    // corrected SIMD unpack (one register per OutReg).
    alignas(32) uint16_t corrections_data[BlockSize] = {0};

    // Pre-compute corrections by walking the gap chain. At each exception
    // position the unpacked value is `gap` (or low-b bits of the value, for
    // the last exception). The correction we add (mod 2^16) is `exc - gap`,
    // so corrected_lane = unpacked + correction = exc.
    const uint16_t *packed_data =
        reinterpret_cast<const uint16_t *>(inputbegin);
    for (size_t idx = start_except_idx; idx != end_except_idx; ++idx) {
      const uint32_t gap =
          read_gap_simd_layout_u16(packed_data, b, next_exception);
      const uint16_t exc_val = static_cast<uint16_t>(except_base[idx]);
      corrections_data[next_exception] =
          static_cast<uint16_t>(exc_val - static_cast<uint16_t>(gap));
      next_exception = next_exception + static_cast<size_t>(gap) + 1;
    }

    unpackblock_corrected(
        inputbegin, outputbegin, b,
        reinterpret_cast<const __m256i *>(corrections_data), sum);
  }

  static inline uint32_t read_gap_simd_layout_u16(const uint16_t *input,
                                                   uint32_t b, size_t index) {
    if (b == 0)
      return 0;
    if (b >= 16)
      return input[index];

    const size_t lane = index & 15;    // 0..15 (which 16-bit lane in __m256i)
    const size_t elem = index >> 4;    // index within that lane stream

    const uint32_t bitpos = static_cast<uint32_t>(elem) * b;
    const uint32_t word = bitpos >> 4;  // 16-bit word index within lane stream
    const uint32_t shift = bitpos & 15;

    const size_t w0 = size_t(word) * 16 + lane;

    uint32_t val = uint32_t(input[w0]) >> shift;

    if (shift + b > 16) {
      const size_t w1 = w0 + 16; // next word in SAME lane (next __m256i)
      val |= uint32_t(input[w1]) << (16 - shift);
    }

    const uint32_t mask = (1u << b) - 1u;
    return val & mask;
  }

  std::string name() const { return "SIMDPFor"; }
};

} // namespace FastPForLib

#endif /* SIMDPFOR_U16_H_ */
