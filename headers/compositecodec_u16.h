/**
 * Composite codec combining SIMDPForU16 + VariableByteU16 for uint16 data.
 *
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 */
#ifndef COMPOSITECODEC_U16_H_
#define COMPOSITECODEC_U16_H_

#include "simdpfor_u16.h"
#include "variablebyte_u16.h"
#include "util.h"

namespace FastPForLib {

class CompositeCodecU16 {
public:
  SIMDPForU16 codec1;
  VariableByteU16 codec2;

  void encodeArray(const uint16_t *in, const size_t length, uint32_t *out,
                   size_t &nvalue) {
    if (nvalue == 0)
      return;
    const size_t roundedlength =
        length / SIMDPForU16::BlockSize * SIMDPForU16::BlockSize;
    size_t nvalue1 = nvalue;
    codec1.encodeArray(in, roundedlength, out, nvalue1);

    if (roundedlength < length) {
      size_t nvalue2 = nvalue - nvalue1;
      codec2.encodeArray(in + roundedlength, length - roundedlength,
                         out + nvalue1, nvalue2);
      nvalue = nvalue1 + nvalue2;
    } else {
      nvalue = nvalue1;
    }
  }

  const uint32_t *decodeArray(const uint32_t *in, const size_t length,
                              uint16_t *out, size_t &nvalue) {
    if (nvalue == 0)
      return in;
    uint16_t *initout(out);
    size_t mynvalue1 = nvalue;
    const uint32_t *in2 = codec1.decodeArray(in, length, out, mynvalue1);
    if (length + in > in2) {
      // SIMDPFor stores sum in out[mynvalue1] and out[mynvalue1+1]
      uint32_t sum1 = static_cast<uint32_t>(initout[mynvalue1]) |
                      (static_cast<uint32_t>(initout[mynvalue1 + 1]) << 16);
      size_t nvalue2 = nvalue - mynvalue1;
      uint16_t *initout2 = out + mynvalue1;
      const uint32_t *in3 =
          codec2.decodeArray(in2, length - (in2 - in), initout2, nvalue2);
      nvalue = mynvalue1 + nvalue2;
      // VariableByte stores sum in initout2[nvalue2] and initout2[nvalue2+1]
      uint32_t sum2 = static_cast<uint32_t>(initout2[nvalue2]) |
                      (static_cast<uint32_t>(initout2[nvalue2 + 1]) << 16);

      uint32_t sum = sum1 + sum2;
      initout[nvalue] = static_cast<uint16_t>(sum & 0xFFFF);
      initout[nvalue + 1] = static_cast<uint16_t>(sum >> 16);
      return in3;
    }
    nvalue = mynvalue1;
    return in2;
  }

  std::string name() const { return codec1.name() + "+" + codec2.name(); }
};

} // namespace FastPForLib

#endif /* COMPOSITECODEC_U16_H_ */
