/**
 * VariableByte encoding for uint16 values with fused sum aggregation.
 *
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 */

#ifndef VARIABLEBYTE_U16_H_
#define VARIABLEBYTE_U16_H_

#include "common.h"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

namespace FastPForLib {

class VariableByteU16 {
public:
  enum { BlockSize = 1 };

  void encodeArray(const uint16_t *in, const size_t length, uint32_t *out,
                   size_t &nvalue) {
    uint8_t *bout = reinterpret_cast<uint8_t *>(out);
    const uint8_t *const initbout = bout;
    size_t bytenvalue = nvalue * sizeof(uint32_t);
    encodeToByteArray(in, length, bout, bytenvalue);
    bout += bytenvalue;
    while (needPaddingTo32Bits(bout)) {
      *bout++ = 0;
    }
    const size_t storageinbytes = bout - initbout;
    assert((storageinbytes % 4) == 0);
    nvalue = storageinbytes / 4;
  }

  void encodeToByteArray(const uint16_t *in, const size_t length,
                         uint8_t *bout, size_t &nvalue) {
    const uint8_t *const initbout = bout;
    for (size_t k = 0; k < length; ++k) {
      const uint16_t val = in[k];
      // uint16 max = 65535, needs at most 3 VByte bytes
      if (val < (1U << 7)) {
        *bout++ = static_cast<uint8_t>(val | 0x80);
      } else if (val < (1U << 14)) {
        *bout++ = static_cast<uint8_t>(val & 0x7F);
        *bout++ = static_cast<uint8_t>((val >> 7) | 0x80);
      } else {
        *bout++ = static_cast<uint8_t>(val & 0x7F);
        *bout++ = static_cast<uint8_t>((val >> 7) & 0x7F);
        *bout++ = static_cast<uint8_t>((val >> 14) | 0x80);
      }
    }
    nvalue = bout - initbout;
  }

  const uint32_t *decodeArray(const uint32_t *in, const size_t length,
                              uint16_t *out, size_t &nvalue) {
    const uint8_t *inbyte = reinterpret_cast<const uint8_t *>(in);
    const size_t bytelen = length * sizeof(uint32_t);
    decodeFromByteArray(inbyte, bytelen, out, nvalue);
    return in + length;
  }

  const uint8_t *decodeFromByteArray(const uint8_t *inbyte,
                                     const size_t length, uint16_t *out,
                                     size_t &nvalue) {
    if (length == 0) {
      nvalue = 0;
      return inbyte;
    }
    const uint8_t *const endbyte = inbyte + length;
    uint16_t *initout = out;

    uint32_t sum = 0;
    uint32_t nv = 0;

    while (endbyte > inbyte + 3) {
      uint8_t c;
      uint16_t v;

      c = inbyte[0];
      v = c & 0x7F;
      if (c >= 128) {
        inbyte += 1;
        nv++;
        sum += v;
        continue;
      }

      c = inbyte[1];
      v |= static_cast<uint16_t>(c & 0x7F) << 7;
      if (c >= 128) {
        inbyte += 2;
        nv++;
        sum += v;
        continue;
      }

      c = inbyte[2];
      inbyte += 3;
      v |= static_cast<uint16_t>(c & 0x03) << 14;
      nv++;
      sum += v;
    }

    while (endbyte > inbyte) {
      unsigned int shift = 0;
      for (uint16_t v = 0; endbyte > inbyte; shift += 7) {
        uint8_t c = *inbyte++;
        v += static_cast<uint16_t>((c & 127) << shift);
        if (c & 128) {
          sum += v;
          nv++;
          break;
        }
      }
    }
    nvalue = nv;

    // store fused sum as uint32 in 2 uint16 slots
    initout[nvalue] = static_cast<uint16_t>(sum & 0xFFFF);
    initout[nvalue + 1] = static_cast<uint16_t>(sum >> 16);

    return inbyte;
  }

  // ── Delta variants ─────────────────────────────────────────────────────────
  //
  // The VB-encoded stream here contains zigzag-encoded deltas. The two methods
  // below decode VByte values, un-zigzag, and un-delta on the fly to recover
  // the ORIGINAL uint16 values, and write the sum of those original values to
  // the overflow slots (same layout as decodeArray).
  //
  // decodeArrayDeltaLocal: prev resets to 0 every 16 decoded elements (the
  //   element at offset 0, 16, 32, ... is an anchor with zigzag-encoded
  //   delta-from-0).
  // decodeArrayDeltaCarry: prev starts at seed_prev (carry from a preceding
  //   SIMDPFor decode); never resets.

  const uint32_t *decodeArrayDeltaLocal(const uint32_t *in, const size_t length,
                                         uint16_t *out, size_t &nvalue) {
    const uint8_t *inbyte = reinterpret_cast<const uint8_t *>(in);
    const size_t bytelen = length * sizeof(uint32_t);
    decodeFromByteArrayDeltaLocal(inbyte, bytelen, out, nvalue);
    return in + length;
  }

  const uint8_t *decodeFromByteArrayDeltaLocal(const uint8_t *inbyte,
                                               const size_t length,
                                               uint16_t *out, size_t &nvalue) {
    if (length == 0) {
      nvalue = 0;
      return inbyte;
    }
    const uint8_t *const endbyte = inbyte + length;
    uint16_t *initout = out;

    uint32_t sum = 0;
    uint32_t nv = 0;
    uint16_t prev = 0;

    // Lambda to consume one VByte value into `v`. Returns true if a value was
    // produced, false if the byte stream ended (only possible mid-padding).
    auto consume_one = [&](uint16_t &v_out) -> bool {
      uint16_t v = 0;
      unsigned int shift = 0;
      while (endbyte > inbyte) {
        uint8_t c = *inbyte++;
        v += static_cast<uint16_t>((c & 127) << shift);
        if (c & 128) {
          v_out = v;
          return true;
        }
        shift += 7;
      }
      return false;
    };

    while (endbyte > inbyte) {
      uint16_t encoded;
      if (!consume_one(encoded)) break;
      const uint16_t delta = ZigzagDec16_inline(encoded);
      uint16_t cur;
      if ((nv & 15) == 0) {
        // Anchor: prev resets to 0, so cur = 0 + delta = delta.
        cur = delta;
      } else {
        cur = static_cast<uint16_t>(prev + delta);
      }
      prev = cur;
      sum += cur;
      ++nv;
    }
    nvalue = nv;

    initout[nvalue] = static_cast<uint16_t>(sum & 0xFFFF);
    initout[nvalue + 1] = static_cast<uint16_t>(sum >> 16);

    return inbyte;
  }

  const uint32_t *decodeArrayDeltaCarry(const uint32_t *in, const size_t length,
                                         uint16_t *out, size_t &nvalue,
                                         uint16_t seed_prev) {
    const uint8_t *inbyte = reinterpret_cast<const uint8_t *>(in);
    const size_t bytelen = length * sizeof(uint32_t);
    decodeFromByteArrayDeltaCarry(inbyte, bytelen, out, nvalue, seed_prev);
    return in + length;
  }

  const uint8_t *decodeFromByteArrayDeltaCarry(const uint8_t *inbyte,
                                               const size_t length,
                                               uint16_t *out, size_t &nvalue,
                                               uint16_t seed_prev) {
    if (length == 0) {
      nvalue = 0;
      return inbyte;
    }
    const uint8_t *const endbyte = inbyte + length;
    uint16_t *initout = out;

    uint32_t sum = 0;
    uint32_t nv = 0;
    uint16_t prev = seed_prev;

    auto consume_one = [&](uint16_t &v_out) -> bool {
      uint16_t v = 0;
      unsigned int shift = 0;
      while (endbyte > inbyte) {
        uint8_t c = *inbyte++;
        v += static_cast<uint16_t>((c & 127) << shift);
        if (c & 128) {
          v_out = v;
          return true;
        }
        shift += 7;
      }
      return false;
    };

    while (endbyte > inbyte) {
      uint16_t encoded;
      if (!consume_one(encoded)) break;
      const uint16_t delta = ZigzagDec16_inline(encoded);
      const uint16_t cur = static_cast<uint16_t>(prev + delta);
      prev = cur;
      sum += cur;
      ++nv;
    }
    nvalue = nv;

    initout[nvalue] = static_cast<uint16_t>(sum & 0xFFFF);
    initout[nvalue + 1] = static_cast<uint16_t>(sum >> 16);

    return inbyte;
  }

  // Local copy of ZigzagDec16 to avoid pulling in the project-level header
  // from inside the external/ tree. Bijective inverse of zigzag-encoding;
  // matches src/codecs/uint16/predictive_codecs_u16.h::ZigzagDec16.
  static inline uint16_t ZigzagDec16_inline(uint16_t z) {
    return static_cast<uint16_t>(
        (z >> 1) ^ static_cast<uint16_t>(-(static_cast<int>(z) & 1)));
  }

  std::string name() const { return "VariableByte"; }
};

} // namespace FastPForLib

#endif /* VARIABLEBYTE_U16_H_ */
