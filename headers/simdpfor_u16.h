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
  size_t maxChunkSize_;
  double exceptionPenalty_;
  size_t total_exceptions_encoded_ = 0;
  size_t total_blocks_encoded_     = 0;

  SIMDPForU16(size_t maxChunkSize = (1U << (32 - blocksizeinbits - 1)),
              double exceptionPenalty = 16.0)
      : codedcopy(BlockSize), miss(BlockSize),
        maxChunkSize_(maxChunkSize), exceptionPenalty_(exceptionPenalty) {}

  double MeanExceptionsPerBlock() const {
    return total_blocks_encoded_ > 0
        ? static_cast<double>(total_exceptions_encoded_) / total_blocks_encoded_
        : 0.0;
  }

  static uint32_t determineBestBase(const DATATYPE *in, size_t size,
                                    double penalty = 16.0) {
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
        double altErate = (Erate * 128 - 1) /
                          (Erate * (1U << b));
        if (altErate > Erate)
          Erate = altErate;
      }
      const double thiscost = b + Erate * penalty;
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

  void unpackblock_corrected_uniform(const uint32_t *source, uint16_t *out,
                                      const uint32_t bit, __m256i anchor,
                                      __m256i *sum) {
    (void)out;
    usimdunpack_u16_corrected_uniform(
        reinterpret_cast<const __m256i *>(source), out, bit, anchor, sum);
  }

  void unpackblock_corrected_delta_local(const uint32_t *source, uint16_t *out,
                                          const uint32_t bit,
                                          const __m256i *corrections,
                                          __m256i *sum) {
    (void)out;
    usimdunpack_u16_corrected_delta_local(
        reinterpret_cast<const __m256i *>(source), out, bit, corrections, sum);
  }

  void unpackblock_corrected_delta_carry(const uint32_t *source, uint16_t *out,
                                          const uint32_t bit,
                                          const __m256i *corrections,
                                          __m256i *carry, __m256i *sum) {
    (void)out;
    usimdunpack_u16_corrected_delta_carry(
        reinterpret_cast<const __m256i *>(source), out, bit, corrections, carry,
        sum);
  }

  // Sub-block pack/unpack wrappers dispatching to the nW primitives.
  void packblock_nW(const uint16_t *source, uint32_t *out, const uint32_t bit,
                    size_t W) {
    auto *o = reinterpret_cast<__m256i *>(out);
    switch (W) {
    case 32:  usimdpack_u16_n32(source, o, bit); break;
    case 64:  usimdpack_u16_n64(source, o, bit); break;
    case 128: usimdpack_u16_n128(source, o, bit); break;
    default:  usimdpack_u16(source, o, bit); break;
    }
  }

  void unpackblock_corrected_uniform_nW(const uint32_t *source, uint16_t *out,
                                         const uint32_t bit, __m256i anchor,
                                         __m256i *sum, size_t W) {
    (void)out;
    const auto *s = reinterpret_cast<const __m256i *>(source);
    switch (W) {
    case 32:  usimdunpack_u16_corrected_uniform_n32(s, out, bit, anchor, sum); break;
    case 64:  usimdunpack_u16_corrected_uniform_n64(s, out, bit, anchor, sum); break;
    case 128: usimdunpack_u16_corrected_uniform_n128(s, out, bit, anchor, sum); break;
    default:  usimdunpack_u16_corrected_uniform(s, out, bit, anchor, sum); break;
    }
  }

  void unpackblock_corrected_nW(const uint32_t *source, uint16_t *out,
                                  const uint32_t bit,
                                  const __m256i *corrections, __m256i *sum,
                                  size_t W) {
    (void)out;
    const auto *s = reinterpret_cast<const __m256i *>(source);
    switch (W) {
    case 32:  usimdunpack_u16_corrected_n32(s, out, bit, corrections, sum); break;
    case 64:  usimdunpack_u16_corrected_n64(s, out, bit, corrections, sum); break;
    case 128: usimdunpack_u16_corrected_n128(s, out, bit, corrections, sum); break;
    default:  usimdunpack_u16_corrected(s, out, bit, corrections, sum); break;
    }
  }

  // W-parametrized block compressor. W must be a multiple of 16 and <= BlockSize.
  uint32_t compressblockPFOR_nW(const DATATYPE *__restrict__ in,
                                 uint32_t *__restrict__ outputbegin,
                                 const uint32_t b,
                                 DATATYPE *__restrict__ &exceptions,
                                 size_t W) {
    if (b == 16) {
      const uint16_t *src = in;
      for (size_t k = 0; k < W / 2; ++k)
        outputbegin[k] = static_cast<uint32_t>(src[2 * k]) |
                         (static_cast<uint32_t>(src[2 * k + 1]) << 16);
      return static_cast<uint32_t>(W);
    }
    size_t exceptcounter = 0;
    const uint32_t maxgap = 1U << b;
    for (uint32_t k = 0; k < W; ++k) {
      miss[exceptcounter] = k;
      exceptcounter += (in[k] >= maxgap);
    }
    if (exceptcounter == 0) {
      packblock_nW(in, outputbegin, b, W);
      return static_cast<uint32_t>(W);
    }
    codedcopy.assign(in, in + W);
    uint32_t firstexcept = miss[0];
    uint32_t prev = 0;
    *(exceptions++) = codedcopy[firstexcept];
    prev = firstexcept;
    if (maxgap < W) {
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
    packblock_nW(&codedcopy[0], outputbegin, b, W);
    return firstexcept;
  }

  // W-parametrized FoR-corrected block decompressor.
  void uncompressblockPFOR_u16_corrected_for_nW(
      const uint32_t *__restrict__ inputbegin,
      uint16_t *__restrict__ outputbegin, const uint32_t b,
      const uint32_t *__restrict__ except_base, size_t start_except_idx,
      size_t end_except_idx, size_t next_exception,
      uint16_t anchor, __m256i *sum, size_t W) {

    const __m256i anchor_bcast = _mm256_set1_epi16(static_cast<short>(anchor));

    if (b == 16) {
      const uint16_t *raw = reinterpret_cast<const uint16_t *>(inputbegin);
      const __m256i zero = _mm256_setzero_si256();
      for (size_t i = 0; i < W; i += 16) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(raw + i));
        v = _mm256_add_epi16(v, anchor_bcast);
        *sum = _mm256_add_epi32(*sum, _mm256_unpacklo_epi16(v, zero));
        *sum = _mm256_add_epi32(*sum, _mm256_unpackhi_epi16(v, zero));
      }
      return;
    }

    if (start_except_idx == end_except_idx) {
      unpackblock_corrected_uniform_nW(inputbegin, outputbegin, b,
                                       anchor_bcast, sum, W);
      return;
    }

    // Exception path: stack-allocate W-element corrections array.
    const size_t n_outreg = W / PACKSIZE;
    alignas(32) uint16_t corrections_data[BlockSize];  // BlockSize upper-bounds W
    {
      __m256i *cd = reinterpret_cast<__m256i *>(corrections_data);
      for (size_t r = 0; r < n_outreg; ++r)
        _mm256_store_si256(cd + r, anchor_bcast);
    }
    const uint16_t *exc16 = reinterpret_cast<const uint16_t *>(except_base);
    const uint16_t *packed_data = reinterpret_cast<const uint16_t *>(inputbegin);
    for (size_t idx = start_except_idx; idx != end_except_idx; ++idx) {
      const uint32_t gap =
          read_gap_simd_layout_u16(packed_data, b, next_exception);
      const uint16_t exc_val = exc16[idx];
      corrections_data[next_exception] +=
          static_cast<uint16_t>(exc_val - static_cast<uint16_t>(gap));
      next_exception = next_exception + static_cast<size_t>(gap) + 1;
    }
    unpackblock_corrected_nW(
        inputbegin, outputbegin, b,
        reinterpret_cast<const __m256i *>(corrections_data), sum, W);
  }

  void encodeArray(const uint16_t *in, const size_t len, uint32_t *out,
                   size_t &nvalue) {
    total_exceptions_encoded_ = 0;
    total_blocks_encoded_     = 0;
    *out++ = static_cast<uint32_t>(len);
    const size_t maxsize = maxChunkSize_;
    size_t totalnvalue(1);
    for (size_t j = 0; j < (len + maxsize - 1U) / maxsize; ++j) {
      size_t i = j * maxsize;
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

  // Corrected + LOCAL delta variant: each OutReg is an independent prefix-sum
  // window (lane 0 = zigzag(in[16v]); lanes 1..15 = zigzag(in[16v+j]-in[16v+j-1])).
  // No inter-OutReg carry.
  const uint32_t *decodeArrayCorrectedDeltaLocal(const uint32_t *in,
                                                  const size_t len,
                                                  uint16_t *out,
                                                  size_t &nvalue) {
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
      in = __decodeArrayCorrectedDeltaLocal(in, finalin - in, out, thisnvalue,
                                             &sum);
      out += thisnvalue;
      totalnvalue += thisnvalue;
    }
    nvalue = totalnvalue;

    __m128i lo = _mm256_castsi256_si128(sum);
    __m128i hi = _mm256_extracti128_si256(sum, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    const auto out_sum = static_cast<uint32_t>(_mm_cvtsi128_si32(s));

    initout[nvalue] = static_cast<uint16_t>(out_sum & 0xFFFF);
    initout[nvalue + 1] = static_cast<uint16_t>(out_sum >> 16);

    return in;
  }

  // Corrected + CARRY delta variant: a single running prefix-sum chain across
  // all OutRegs / blocks. `last_value_out` receives the final decoded value
  // (= in[nvalue-1]) so a VB tail can seed its scalar prev. Initial carry = 0.
  const uint32_t *decodeArrayCorrectedDeltaCarry(const uint32_t *in,
                                                  const size_t len,
                                                  uint16_t *out,
                                                  size_t &nvalue,
                                                  uint16_t *last_value_out) {
    nvalue = *in++;
    if (nvalue == 0) {
      if (last_value_out)
        *last_value_out = 0;
      return in;
    }
    const uint32_t *const finalin = in + len;
    size_t totalnvalue(0);
    __m256i sum = _mm256_setzero_si256();
    __m256i carry = _mm256_setzero_si256();
    uint16_t *initout = out;
    while (totalnvalue < nvalue) {
      size_t thisnvalue = nvalue - totalnvalue;
      in = __decodeArrayCorrectedDeltaCarry(in, finalin - in, out, thisnvalue,
                                             &carry, &sum);
      out += thisnvalue;
      totalnvalue += thisnvalue;
    }
    nvalue = totalnvalue;

    __m128i lo = _mm256_castsi256_si128(sum);
    __m128i hi = _mm256_extracti128_si256(sum, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    const auto out_sum = static_cast<uint32_t>(_mm_cvtsi128_si32(s));

    initout[nvalue] = static_cast<uint16_t>(out_sum & 0xFFFF);
    initout[nvalue + 1] = static_cast<uint16_t>(out_sum >> 16);

    if (last_value_out) {
      // carry holds broadcast(in[nvalue-1]); extract lane 0.
      *last_value_out =
          static_cast<uint16_t>(_mm256_extract_epi16(carry, 0));
    }

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

  // ── FoR-global helpers ──────────────────────────────────────────────────────

  // Per-block helper for FoR-global decode. The encoder stored residuals =
  // (original - anchor) for each block; the decoder adds anchor back via the
  // corrections array. Pre-fills corrections with anchor broadcast, then adds
  // (exc_val - gap) at exception positions so the SIMD add reconstructs the
  // original values. b==16 is handled separately (raw residuals; no gap chain).
  void uncompressblockPFOR_u16_corrected_for(
      const uint32_t *__restrict__ inputbegin,
      uint16_t *__restrict__ outputbegin, const uint32_t b,
      const uint32_t *__restrict__ except_base, size_t start_except_idx,
      size_t end_except_idx, size_t next_exception,
      uint16_t anchor, __m256i *sum) {

    const __m256i anchor_bcast = _mm256_set1_epi16(static_cast<short>(anchor));

    if (b == 16) {
      // Raw residuals, no gap-chain. Add anchor to each value before
      // aggregating: residual + anchor = original (all mod 2^16).
      const uint16_t *raw = reinterpret_cast<const uint16_t *>(inputbegin);
      const __m256i zero = _mm256_setzero_si256();
      for (size_t i = 0; i < BlockSize; i += 16) {
        __m256i v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(raw + i));
        v = _mm256_add_epi16(v, anchor_bcast);
        *sum = _mm256_add_epi32(*sum, _mm256_unpacklo_epi16(v, zero));
        *sum = _mm256_add_epi32(*sum, _mm256_unpackhi_epi16(v, zero));
      }
      return;
    }

    if (start_except_idx == end_except_idx) {
      // Exception-free (common for FoR): uniform anchor, no corrections array.
      unpackblock_corrected_uniform(inputbegin, outputbegin, b,
                                     anchor_bcast, sum);
      return;
    }

    // With exceptions (rare for FoR): pre-fill corrections with anchor, then
    // += (exc_val - gap) at exception positions.
    alignas(32) uint16_t corrections_data[BlockSize];
    {
      __m256i *cd = reinterpret_cast<__m256i *>(corrections_data);
      for (int r = 0; r < BlockSizeInUnitsOfPackSize; ++r)
        _mm256_store_si256(cd + r, anchor_bcast);
    }
    const uint16_t *packed_data =
        reinterpret_cast<const uint16_t *>(inputbegin);
    for (size_t idx = start_except_idx; idx != end_except_idx; ++idx) {
      const uint32_t gap =
          read_gap_simd_layout_u16(packed_data, b, next_exception);
      const uint16_t exc_val = static_cast<uint16_t>(except_base[idx]);
      corrections_data[next_exception] +=
          static_cast<uint16_t>(exc_val - static_cast<uint16_t>(gap));
      next_exception = next_exception + static_cast<size_t>(gap) + 1;
    }
    unpackblock_corrected(
        inputbegin, outputbegin, b,
        reinterpret_cast<const __m256i *>(corrections_data), sum);
  }

  // Inner decode loop for FoR-global. block_anchors[k] is the per-block anchor
  // for block k within this __encodeArray chunk.
  const uint32_t *__decodeArrayCorrectedFor(const uint32_t *in,
                                             const size_t len, uint16_t *out,
                                             size_t &nvalue,
                                             const uint16_t *block_anchors,
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

      uncompressblockPFOR_u16_corrected_for(in, out, b, except, except_offset,
                                             end_except_idx, firstexcept,
                                             block_anchors[k], sum);
      except_offset = end_except_idx;
      in += (BlockSize * b) / 32;
      out += BlockSize;
    }

    return except + except_offset;
  }

  // Outer decode entry for FoR-global. `anchors` is an array of per-block
  // (256-element) anchor values, indexed globally across all inner chunks.
  const uint32_t *decodeArrayCorrectedFor(const uint32_t *in,
                                           const size_t len, uint16_t *out,
                                           size_t &nvalue,
                                           const uint16_t *anchors) {
    nvalue = *in++;
    if (nvalue == 0) {
      return in;
    }
    const uint32_t *const finalin = in + len;
    size_t totalnvalue(0);
    __m256i sum = _mm256_setzero_si256();
    uint16_t *initout = out;
    size_t anchor_idx = 0;

    while (totalnvalue < nvalue) {
      size_t thisnvalue = nvalue - totalnvalue;
      in = __decodeArrayCorrectedFor(in, finalin - in, out, thisnvalue,
                                      anchors + anchor_idx, &sum);
      anchor_idx += thisnvalue / BlockSize;
      out += thisnvalue;
      totalnvalue += thisnvalue;
    }
    nvalue = totalnvalue;

    __m128i lo = _mm256_castsi256_si128(sum);
    __m128i hi = _mm256_extracti128_si256(sum, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    const auto out_sum = static_cast<uint32_t>(_mm_cvtsi128_si32(s));

    initout[nvalue] = static_cast<uint16_t>(out_sum & 0xFFFF);
    initout[nvalue + 1] = static_cast<uint16_t>(out_sum >> 16);

    return in;
  }

  // ── Flat-format encode/decode ────────────────────────────────────────────────
  //
  // Format: [n_blocks:u32][bs:u8×n_blocks padded to 4B][headers:u32×n_blocks]
  //         [payloads:sum(bs[k]×BlockSize/32) u32 words][exceptions:u32 per val]
  //
  // All per-block b values are in a flat sequential array; the decode loop reads
  // bs[k] at a known address (hardware-prefetchable sequential array) rather than
  // from a data-dependent stream position. Breaks the per-block pointer chain that
  // makes adaptive_b ~1.4× slower than global_b despite identical compression.

  void encodeArrayFlat(const uint16_t *in, const size_t len, uint32_t *out,
                       size_t &nvalue, size_t window_size = BlockSize) {
    assert(len % window_size == 0);
    const size_t n_blocks = len / window_size;
    const uint32_t *const initout = out;

    // Step 1: compute per-block bs[] (one determineBestBase call per block)
    // bs_buf sized for smallest window (W=32 with 65536 elements = 2048 blocks).
    uint8_t bs_buf[2048];
    for (size_t k = 0; k < n_blocks; ++k)
      bs_buf[k] = static_cast<uint8_t>(
          determineBestBase(in + k * window_size, window_size, exceptionPenalty_));

    // Step 2: total payload words — needed to locate exception area upfront
    // Payload for W-element block at b bits: ceil(W*b/256) * 8 uint32 words.
    size_t total_payload_words = 0;
    for (size_t k = 0; k < n_blocks; ++k)
      total_payload_words += ((window_size * static_cast<size_t>(bs_buf[k]) + 255) / 256) * 8;

    // Write n_blocks
    *out++ = static_cast<uint32_t>(n_blocks);

    // Write bs[] packed as u8 in u32 words (zero-pad last word)
    const size_t bs_words = (n_blocks + 3) / 4;
    auto *bs_raw = reinterpret_cast<uint8_t *>(out);
    for (size_t j = 0; j < bs_words * 4; ++j)
      bs_raw[j] = (j < n_blocks) ? bs_buf[j] : 0u;
    out += bs_words;

    // Reserve header words — filled after payload encoding
    uint32_t *headers_ptr = out;
    out += n_blocks;

    // Exception area starts at a known offset past all payloads. Data is uint16,
    // so each exception value is a uint16 — pack them 2 per uint32 word (half the
    // space of one-per-u32). Both flat decoders read them back via a uint16 view.
    uint32_t *const except_area = out + total_payload_words;
    uint16_t *except_out = reinterpret_cast<uint16_t *>(except_area);
    size_t cumulative_exc = 0;

    const uint32_t firstexceptmask = (1U << blocksizeinbits) - 1;
    alignas(32) uint16_t per_block_exc[BlockSize];  // BlockSize upper-bounds W

    for (size_t k = 0; k < n_blocks; ++k) {
      uint16_t *exc_ptr = per_block_exc;
      const uint32_t firstexcept =
          compressblockPFOR_nW(in + k * window_size, out, bs_buf[k], exc_ptr, window_size);
      const size_t block_exc =
          static_cast<size_t>(exc_ptr - per_block_exc);
      out += ((window_size * static_cast<size_t>(bs_buf[k]) + 255) / 256) * 8;
      cumulative_exc += block_exc;
      headers_ptr[k] = (firstexcept & firstexceptmask) |
                       (static_cast<uint32_t>(cumulative_exc) << blocksizeinbits);
      for (size_t e = 0; e < block_exc; ++e)
        *except_out++ = per_block_exc[e];
    }
    // Zero-pad the trailing half-word so the final uint32 is fully defined.
    if (cumulative_exc & 1)
      *except_out = 0;

    total_exceptions_encoded_ += cumulative_exc;
    total_blocks_encoded_     += n_blocks;
    // Exception stream occupies ceil(cumulative_exc / 2) uint32 words.
    nvalue = static_cast<size_t>(except_area - initout) + (cumulative_exc + 1) / 2;
  }

  // Flat-format decode for corrected (non-FoR) path. window_size must match
  // the value passed to encodeArrayFlat.
  const uint32_t *decodeArrayFlatCorrected(const uint32_t *in,
                                            const size_t /*len*/,
                                            uint16_t *out, size_t &nvalue,
                                            size_t window_size = BlockSize) {
    uint16_t *initout = out;
    __m256i sum = _mm256_setzero_si256();

    const size_t n_blocks = *in++;
    const size_t bs_words = (n_blocks + 3) / 4;
    const uint8_t *bs = reinterpret_cast<const uint8_t *>(in);
    in += bs_words;
    const uint32_t *headers = in;
    in += n_blocks;
    const uint32_t *payload_base = in;

    size_t total_payload_words = 0;
    for (size_t k = 0; k < n_blocks; ++k)
      total_payload_words += ((window_size * static_cast<size_t>(bs[k]) + 255) / 256) * 8;
    const uint32_t *except_base = payload_base + total_payload_words;

    const uint32_t firstexceptmask = (1U << blocksizeinbits) - 1;
    const uint32_t *payload_ptr = payload_base;
    size_t except_offset = 0;

    for (size_t k = 0; k < n_blocks; ++k) {
      const uint32_t b        = bs[k];   // flat array — no data-dep load chain
      const uint32_t header   = headers[k];
      const uint32_t firstexcept = header & firstexceptmask;
      const uint32_t exceptindex = header >> blocksizeinbits;

      uncompressblockPFOR_u16_corrected_nW(payload_ptr, out, b,
                                            except_base, except_offset,
                                            exceptindex, firstexcept, &sum,
                                            window_size);
      except_offset = exceptindex;
      payload_ptr  += ((window_size * b + 255) / 256) * 8;
      out          += window_size;
    }

    nvalue = n_blocks * window_size;

    __m128i lo = _mm256_castsi256_si128(sum);
    __m128i hi = _mm256_extracti128_si256(sum, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    const auto out_sum = static_cast<uint32_t>(_mm_cvtsi128_si32(s));
    initout[nvalue]     = static_cast<uint16_t>(out_sum & 0xFFFF);
    initout[nvalue + 1] = static_cast<uint16_t>(out_sum >> 16);

    // Exceptions are packed 2 uint16 per uint32 word.
    return except_base + (except_offset + 1) / 2;
  }

  // Flat-format decode for FoR-corrected path. anchors[k] is the per-block anchor.
  const uint32_t *decodeArrayFlatCorrectedFor(const uint32_t *in,
                                               const size_t /*len*/,
                                               uint16_t *out, size_t &nvalue,
                                               const uint16_t *anchors,
                                               size_t window_size = BlockSize) {
    uint16_t *initout = out;
    __m256i sum = _mm256_setzero_si256();

    const size_t n_blocks = *in++;
    const size_t bs_words = (n_blocks + 3) / 4;
    const uint8_t *bs = reinterpret_cast<const uint8_t *>(in);
    in += bs_words;
    const uint32_t *headers = in;
    in += n_blocks;
    const uint32_t *payload_base = in;

    size_t total_payload_words = 0;
    for (size_t k = 0; k < n_blocks; ++k)
      total_payload_words += ((window_size * static_cast<size_t>(bs[k]) + 255) / 256) * 8;
    const uint32_t *except_base = payload_base + total_payload_words;

    const uint32_t firstexceptmask = (1U << blocksizeinbits) - 1;
    const uint32_t *payload_ptr = payload_base;
    size_t except_offset = 0;

    for (size_t k = 0; k < n_blocks; ++k) {
      const uint32_t b        = bs[k];   // flat array — no data-dep load chain
      const uint32_t header   = headers[k];
      const uint32_t firstexcept = header & firstexceptmask;
      const uint32_t exceptindex = header >> blocksizeinbits;

      uncompressblockPFOR_u16_corrected_for_nW(payload_ptr, out, b,
                                                except_base, except_offset,
                                                exceptindex, firstexcept,
                                                anchors[k], &sum, window_size);
      except_offset = exceptindex;
      payload_ptr  += ((window_size * b + 255) / 256) * 8;
      out          += window_size;
    }

    nvalue = n_blocks * window_size;

    __m128i lo = _mm256_castsi256_si128(sum);
    __m128i hi = _mm256_extracti128_si256(sum, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    const auto out_sum = static_cast<uint32_t>(_mm_cvtsi128_si32(s));
    initout[nvalue]     = static_cast<uint16_t>(out_sum & 0xFFFF);
    initout[nvalue + 1] = static_cast<uint16_t>(out_sum >> 16);

    // Exceptions are packed 2 uint16 per uint32 word.
    return except_base + (except_offset + 1) / 2;
  }

  void __encodeArray(const uint16_t *in, const size_t len, uint32_t *out,
                     size_t &nvalue) {
    checkifdivisibleby(len, BlockSize);
    const uint32_t *const initout(out);
    std::vector<DATATYPE> exceptions;
    exceptions.resize(len);
    DATATYPE *__restrict__ i = &exceptions[0];
    const uint32_t b = determineBestBase(in, len, exceptionPenalty_);
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
    total_exceptions_encoded_ += howmanyexcept;
    total_blocks_encoded_     += len / BlockSize;
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

  // Same control flow as __decodeArrayCorrected; each block goes through the
  // corrected+delta-local pipeline (zigzag_dec + per-OutReg prefix sum +
  // aggregate). No inter-OutReg carry.
  const uint32_t *__decodeArrayCorrectedDeltaLocal(const uint32_t *in,
                                                    const size_t len,
                                                    uint16_t *out,
                                                    size_t &nvalue,
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

      uncompressblockPFOR_u16_corrected_delta_local(
          in, out, b, except, except_offset, end_except_idx, firstexcept, sum);
      except_offset = end_except_idx;
      in += (BlockSize * b) / 32;
      out += BlockSize;
    }

    return except + except_offset;
  }

  // Same control flow as __decodeArrayCorrected; each block goes through the
  // corrected+delta-carry pipeline. `carry` is a single __m256i broadcast of
  // the most recently decoded value, maintained across blocks.
  const uint32_t *__decodeArrayCorrectedDeltaCarry(const uint32_t *in,
                                                    const size_t len,
                                                    uint16_t *out,
                                                    size_t &nvalue,
                                                    __m256i *carry,
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

      uncompressblockPFOR_u16_corrected_delta_carry(in, out, b, except,
                                                     except_offset,
                                                     end_except_idx,
                                                     firstexcept, carry, sum);
      except_offset = end_except_idx;
      in += (BlockSize * b) / 32;
      out += BlockSize;
    }

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

  // W-parametrized variant of uncompressblockPFOR_u16_corrected for the flat
  // (adaptive_b) decode path. Two differences from the 256-only helper above:
  //   1. Operates on a W-element sub-block (W ∈ {32,64,128,256}).
  //   2. Reads exceptions from a uint16-packed stream (2 per uint32 word) — the
  //      flat encoder packs them this way; the non-flat global_b path does not,
  //      so that path keeps using the helper above.
  void uncompressblockPFOR_u16_corrected_nW(
      const uint32_t *__restrict__ inputbegin,
      uint16_t *__restrict__ outputbegin, const uint32_t b,
      const uint32_t *__restrict__ except_base, size_t start_except_idx,
      size_t end_except_idx, size_t next_exception, __m256i *sum, size_t W) {
    if (b == 16) {
      const uint16_t *raw = reinterpret_cast<const uint16_t *>(inputbegin);
      const __m256i zero = _mm256_setzero_si256();
      for (size_t i = 0; i < W; i += 16) {
        __m256i v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(raw + i));
        *sum = _mm256_add_epi32(*sum, _mm256_unpacklo_epi16(v, zero));
        *sum = _mm256_add_epi32(*sum, _mm256_unpackhi_epi16(v, zero));
      }
      return;
    }

    if (start_except_idx == end_except_idx) {
      // Exception-free block: plain fused unpack. The non-FoR path has no
      // anchor, so use the uniform helper with a zero anchor (adds 0 per lane).
      unpackblock_corrected_uniform_nW(inputbegin, outputbegin, b,
                                       _mm256_setzero_si256(), sum, W);
      return;
    }

    alignas(32) uint16_t corrections_data[BlockSize] = {0};  // BlockSize ⊇ W
    const uint16_t *exc16 = reinterpret_cast<const uint16_t *>(except_base);
    const uint16_t *packed_data =
        reinterpret_cast<const uint16_t *>(inputbegin);
    for (size_t idx = start_except_idx; idx != end_except_idx; ++idx) {
      const uint32_t gap =
          read_gap_simd_layout_u16(packed_data, b, next_exception);
      const uint16_t exc_val = exc16[idx];
      corrections_data[next_exception] =
          static_cast<uint16_t>(exc_val - static_cast<uint16_t>(gap));
      next_exception = next_exception + static_cast<size_t>(gap) + 1;
    }

    unpackblock_corrected_nW(
        inputbegin, outputbegin, b,
        reinterpret_cast<const __m256i *>(corrections_data), sum, W);
  }

  // Per-block helper for delta-local: walks the exception gap chain, writes
  // (exc - gap) into a stack-local corrections array, then runs the
  // corrected+delta-local SIMD pipeline. For b == 16 the corrections array
  // stays all-zero (no exceptions encoded at b == 16); the SIMD pipeline still
  // applies zigzag_dec + prefix_sum + aggregate to the raw loaded data.
  void uncompressblockPFOR_u16_corrected_delta_local(
      const uint32_t *__restrict__ inputbegin,
      uint16_t *__restrict__ outputbegin, const uint32_t b,
      const uint32_t *__restrict__ except_base, size_t start_except_idx,
      size_t end_except_idx, size_t next_exception, __m256i *sum) {
    alignas(32) uint16_t corrections_data[BlockSize] = {0};

    if (b < 16) {
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
    }

    unpackblock_corrected_delta_local(
        inputbegin, outputbegin, b,
        reinterpret_cast<const __m256i *>(corrections_data), sum);
  }

  // Per-block helper for delta-carry: same as delta_local but additionally
  // threads the broadcast-carry __m256i through the SIMD pipeline.
  void uncompressblockPFOR_u16_corrected_delta_carry(
      const uint32_t *__restrict__ inputbegin,
      uint16_t *__restrict__ outputbegin, const uint32_t b,
      const uint32_t *__restrict__ except_base, size_t start_except_idx,
      size_t end_except_idx, size_t next_exception, __m256i *carry,
      __m256i *sum) {
    alignas(32) uint16_t corrections_data[BlockSize] = {0};

    if (b < 16) {
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
    }

    unpackblock_corrected_delta_carry(
        inputbegin, outputbegin, b,
        reinterpret_cast<const __m256i *>(corrections_data), carry, sum);
  }

  static inline uint32_t read_gap_simd_layout_u16(const uint16_t *input,
                                                   uint32_t b, size_t index) {
    if (b == 0)
      return 0;
    if (b >= 16)
      return input[index];

    const size_t lane = index & 15;
    const size_t elem = index >> 4;
    const uint32_t bitpos = static_cast<uint32_t>(elem) * b;
    const uint32_t word = bitpos >> 4;
    const uint32_t shift = bitpos & 15;
    const size_t w0 = size_t(word) * 16 + lane;
    uint32_t val = uint32_t(input[w0]) >> shift;
    if (shift + b > 16) {
      const size_t w1 = w0 + 16;
      val |= uint32_t(input[w1]) << (16 - shift);
    }
    return val & ((1u << b) - 1u);
  }

  std::string name() const { return "SIMDPFor"; }
};

} // namespace FastPForLib

#endif /* SIMDPFOR_U16_H_ */
