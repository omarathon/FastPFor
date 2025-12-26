/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Daniel Lemire, http://lemire.me/en/
 */

#ifndef SIMDPFOR_H_
#define SIMDPFOR_H_

#include "common.h"
#include "codecs.h"
#include "usimdbitpacking_new.h"
#include "usimdbitpacking_new.h"
#include "util.h"
#include <iostream>

namespace FastPForLib {

/**
 * This implements as best as possible the PFor scheme
 * from  Zukowski et al., Super-Scalar RAM-CPU Cache Compression
 * with SIMD acceleration.
 *
 *  Implemented by D. Lemire
 *
 * Small differences:
 *  1. We don't write the exception section is reverse order.
 *     It is written in forward order.
 *
 *  2.  Obviously, the code is specific to 32-bit integers whereas
 *      the original description allowed for different data types.
 *
 *  3. Because we assume pfor delta, we don't compute the base (that is
 *     the frame). Correspondingly, we don't sort a sample. Instead,
 *     we use a fast approach to identify the best number of bits
 *     based on the computation of the integer logarithm.
 *
 *  4. Though it is not clear how the sample is taken in the original
 *     paper, we consider a consecutive sample of up to 64K samples.
 *
 */
class SIMDPFor : public IntegerCODEC {
public:
  using IntegerCODEC::encodeArray;
  using IntegerCODEC::decodeArray;

  enum {
    BlockSizeInUnitsOfPackSize = 4,
    PACKSIZE = 32,
    BlockSize = BlockSizeInUnitsOfPackSize * PACKSIZE,
    blocksizeinbits = 7 // constexprbits(BlockSize)
  };
  // these are reusable buffers
  std::vector<uint32_t> codedcopy;
  std::vector<uint32_t> miss;
  typedef uint32_t
      DATATYPE; // this is so that our code looks more like the original paper

  SIMDPFor() : codedcopy(BlockSize), miss(BlockSize) {}
  // for delta coding, we don't use a base.
  static uint32_t determineBestBase(const DATATYPE *in, size_t size) {
    if (size == 0)
      return 0;
    const size_t defaultsamplesize = 64 * 1024;
    // the original paper describes sorting
    // a sample, but this only makes sense if you
    // are coding a frame of reference.
    size_t samplesize = size > defaultsamplesize ? defaultsamplesize : size;
    uint32_t freqs[33];
    for (uint32_t k = 0; k <= 32; ++k)
      freqs[k] = 0;
    // we choose the sample to be consecutive
    uint32_t rstart =
        size > samplesize
            ? (rand() % (static_cast<uint32_t>(size - samplesize)))
            : 0U;
    for (uint32_t k = rstart; k < rstart + samplesize; ++k) {
      freqs[asmbits(in[k])]++;
    }
    uint32_t bestb = 32;
    uint32_t numberofexceptions = 0;
    double Erate = 0;
    double bestcost = 32;
    for (uint32_t b = bestb - 1; b < 32; --b) {
      numberofexceptions += freqs[b + 1];
      Erate = static_cast<double>(numberofexceptions) /
              static_cast<double>(samplesize);
      /**
      * though this is not explicit in the original paper, you
      * need to somehow compensate for compulsory exceptions
      * when the chosen number of bits is small.
      *
      * We use their formula (3.1.5) to estimate actual number
      * of total exceptions, including compulsory exceptions.
      */
      if (numberofexceptions > 0) {
        double altErate = (Erate * 128 - 1) / (Erate * (1U << b));
        if (altErate > Erate)
          Erate = altErate;
      }
      const double thiscost = b + Erate * 32;
      if (thiscost <= bestcost) {
        bestcost = thiscost;
        bestb = b;
      }
    }
    return bestb;
  }

  // returns location of first exception or BlockSize if there is none
  uint32_t compressblockPFOR(const DATATYPE *__restrict__ in,
                             uint32_t *__restrict__ outputbegin,
                             const uint32_t b,
                             DATATYPE *__restrict__ &exceptions) {
    if (b == 32) {
      for (size_t k = 0; k < BlockSize; ++k)
        *(outputbegin++) = *(in++);
      return BlockSize;
    }
    size_t exceptcounter = 0;
    const uint32_t maxgap = 1U << b;
    {
      std::vector<uint32_t>::iterator cci = codedcopy.begin();
      for (uint32_t k = 0; k < BlockSize; ++k, ++cci) {
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
        // they don't include this part, but it is required:
        while (cur > maxgap + prev) {
          // compulsory exception
          uint32_t compulcur = prev + maxgap;
          *(exceptions++) = codedcopy[compulcur];
          codedcopy[prev] = maxgap - 1;
          prev = compulcur;
        }
        *(exceptions++) = codedcopy[cur];
        codedcopy[prev] = cur - prev - 1;
        prev = cur;
      }
    } else {
      for (uint32_t i = 1; i < exceptcounter; ++i) {
        uint32_t cur = miss[i];
        *(exceptions++) = codedcopy[cur];
        codedcopy[prev] = cur - prev - 1;
        prev = cur;
      }
    }
    packblock(&codedcopy[0], outputbegin, b);
    return firstexcept;
  }

  void packblock(const uint32_t *source, uint32_t *out, const uint32_t bit) {
    usimdpack(source, reinterpret_cast<__m128i *>(out), bit);
  }

  void unpackblock(const uint32_t *source, uint32_t *out, const uint32_t bit, __m128i* sum) {
    usimdunpack_new(reinterpret_cast<const __m128i *>(source), out, bit, sum);
  }

  void encodeArray(const uint32_t *in, const size_t len, uint32_t *out,
                   size_t &nvalue) override {
    *out++ = static_cast<uint32_t>(len);
#ifndef NDEBUG
    const uint32_t *const finalin(in + len);
#endif
    const uint32_t maxsize = (1U << (32 - blocksizeinbits - 1));
    size_t totalnvalue(1);
    // for (size_t i = 0; i < len; i += maxsize)
    for (size_t j = 0; j < (len + maxsize - 1U) / maxsize; ++j) {
      size_t i = j << (32 - blocksizeinbits - 1);
      size_t l = maxsize;
      if (i + maxsize > len) {
        l = len - i;
        assert(l <= maxsize);
      }
      size_t thisnvalue = nvalue - totalnvalue;
      assert(in + i + l <= finalin);
      __encodeArray(&in[i], l, out, thisnvalue);
      totalnvalue += thisnvalue;
      assert(totalnvalue <= nvalue);
      out += thisnvalue;
    }
    nvalue = totalnvalue;
  }
  const uint32_t *decodeArray(const uint32_t *in, const size_t len,
                              uint32_t *out, size_t &nvalue) override {
    nvalue = *in++;
    if (nvalue == 0) {
      return in;
    }
      
#ifndef NDEBUG
    const uint32_t *const initin = in;
#endif
    const uint32_t *const finalin = in + len;
    size_t totalnvalue(0);
    __m128i sum = _mm_setzero_si128();
    int32_t delta_sum = 0;
    uint32_t* initout = out;
    while (totalnvalue < nvalue) {
      size_t thisnvalue = nvalue - totalnvalue;
#ifndef NDEBUG
      const uint32_t *const befin(in);
#endif
      assert(finalin <= len + in);
      in = __decodeArray(in, finalin - in, out, thisnvalue, &sum, &delta_sum);
      assert(in > befin);
      assert(in <= finalin);
      out += thisnvalue;
      totalnvalue += thisnvalue;
      assert(totalnvalue <= nvalue);
    }
    assert(in <= len + initin);
    assert(in <= finalin);
    nvalue = totalnvalue;

    __m128i s = sum;
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1,0,3,2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2,3,0,1)));
    const auto out_sum = (uint32_t)(_mm_cvtsi128_si32(s) + delta_sum /* correct exceptions */);
    initout[nvalue] = out_sum;

    return in;
  }

  void __encodeArray(const uint32_t *in, const size_t len, uint32_t *out,
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
    const size_t howmanyexcept = i - &exceptions[0];
    for (uint32_t t = 0; t < howmanyexcept; ++t)
      *out++ = exceptions[t];
    nvalue = out - initout;
  }

#ifndef NDEBUG
  const uint32_t *__decodeArray(const uint32_t *in, const size_t len,
#else
  const uint32_t *__decodeArray(const uint32_t *in, const size_t,
#endif
                                uint32_t *out, size_t &nvalue, __m128i* sum, int32_t* delta_sum) {
#ifndef NDEBUG
    const uint32_t *const initin(in);
#endif
    nvalue = *in++;
    checkifdivisibleby(nvalue, BlockSize);
    const uint32_t b = *in++;
    const DATATYPE *__restrict__ except =
        in + nvalue * b / 32 + nvalue / BlockSize;
    const uint32_t bitsforfirstexcept = blocksizeinbits;
    const uint32_t firstexceptmask = (1U << blocksizeinbits) - 1;
    const DATATYPE *endexceptpointer = except;
    const DATATYPE *const initexcept(except);
    for (size_t k = 0; k < nvalue / BlockSize; ++k) {
      const uint32_t *const headerin(in);
      ++in;
      const uint32_t firstexcept = *headerin & firstexceptmask;
      const uint32_t exceptindex = *headerin >> bitsforfirstexcept;
      endexceptpointer = initexcept + exceptindex;
      uncompressblockPFOR(in, out, b, except, endexceptpointer, firstexcept, sum, delta_sum);
      in += (BlockSize * b) / 32;
      out += BlockSize;
    }
    assert(initin + len >= in);
    assert(initin + len >= endexceptpointer);
    return endexceptpointer;
  }

  static inline uint32_t read_gap(
      const uint32_t* input,
      uint32_t b,
      size_t index)
  {
      if (b == 0) return 0;
      if (b >= 32) {
          return input[index];
      }

      const uint64_t bitpos = uint64_t(index) * b;
      const uint32_t word = uint32_t(bitpos >> 5); // / 32
      const uint32_t shift = uint32_t(bitpos & 31);

      // Always load safely
      uint64_t val = uint64_t(input[word]) >> shift;

      if (shift + b > 32) {
          // Safe: shift is in [1,31], so (32-shift) is in [1,31]
          val |= uint64_t(input[word + 1]) << (32 - shift);
      }

      // Mask without UB
      const uint32_t mask = (uint32_t(1) << b) - 1;
      return uint32_t(val) & mask;
  }

  void uncompressblockPFOR(
    const uint32_t
      *__restrict__ inputbegin, // points to the first packed word
    DATATYPE *__restrict__ outputbegin,
    const uint32_t b,
    const DATATYPE *__restrict__
        &i, // i points to value of the first exception
    const DATATYPE *__restrict__ end_exception,
    size_t next_exception // points to the position of the first exception
    , __m128i* sum, int32_t* delta_sum)
  {
    // decode block
    unpackblock(inputbegin, reinterpret_cast<uint32_t *> (outputbegin), b, sum);

    for (size_t cur = next_exception; i != end_exception;
         cur = next_exception) {
      const auto gap = cur < BlockSize ? read_gap(inputbegin, b, cur) : 0;
      std::cout << "gap " << gap << " exc " << *i << std::endl;
      next_exception = cur + static_cast<size_t>(gap) + 1;
      *delta_sum += (-gap + (*i));
      i++;
    }

    // while (i != end_exception) {
    //     // Read gap directly from packed stream
    //     const uint32_t gap = read_gap(inputbegin, b, next_exception);

    //     // Correct sum
    //     *delta_sum += int32_t(*i) - int32_t(gap);

    //     // Advance to next exception
    //     next_exception += size_t(gap) + 1;
    //     ++i;

    //     // __builtin_prefetch(
    //     //     inputbegin + ((uint64_t(next_exception) * b) >> 5),
    //     //     0,  // read
    //     //     1   // low temporal locality
    //     // );
    // }
  }

  virtual std::string name() const override {
    return "SIMDPFor";
  }
};

} // namespace FastPForLib

#endif /* SIMDPFOR_H_ */
