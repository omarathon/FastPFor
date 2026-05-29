/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 */
#ifndef USIMDBITPACKING_U16_H_
#define USIMDBITPACKING_U16_H_

#include "common.h"

namespace FastPForLib {

void usimdpack_u16(const uint16_t *__restrict__ in, __m256i *__restrict__ out,
                   uint32_t bit);

void usimdunpack_u16(const __m256i *__restrict__ in, uint16_t *__restrict__ out,
                     uint32_t bit, __m256i *__restrict__ sum);

// Corrected variant: per-OutReg `corrections` (one __m256i per 16-lane OutReg,
// BlockSize/16 entries) are added (uint16) to OutReg before sum aggregation,
// folding exception correction into the hot loop. `out` is unused (phantom).
void usimdunpack_u16_corrected(const __m256i *__restrict__ in,
                               uint16_t *__restrict__ out, uint32_t bit,
                               const __m256i *__restrict__ corrections,
                               __m256i *__restrict__ sum);

// Uniform-corrected variant: like corrected but takes a single broadcast
// `anchor` applied to every OutReg. Used by FoR-global exception-free path —
// eliminates the 512-byte corrections array and 16-store fill per block.
void usimdunpack_u16_corrected_uniform(const __m256i *__restrict__ in,
                                        uint16_t *__restrict__ out,
                                        uint32_t bit,
                                        __m256i anchor,
                                        __m256i *__restrict__ sum);

// Corrected + LOCAL delta variant: per-OutReg pipeline is correction → zigzag
// decode → per-OutReg prefix sum → aggregate. Each OutReg holds 16 consecutive
// elements; lane 0 is a zigzag-encoded anchor (delta from 0), lanes 1..15 are
// zigzag-encoded deltas from lane j-1. No inter-OutReg carry.
void usimdunpack_u16_corrected_delta_local(const __m256i *__restrict__ in,
                                            uint16_t *__restrict__ out,
                                            uint32_t bit,
                                            const __m256i *__restrict__ corrections,
                                            __m256i *__restrict__ sum);

// Corrected + CARRY delta variant: same pipeline as local but additionally adds
// `*carry` (prev OutReg's last decoded value broadcast to all 16 lanes) before
// aggregation, and updates `*carry` to the new lane-15 broadcast. `*carry` must
// be initialized by the caller (e.g. zeroed for first OutReg of stream).
void usimdunpack_u16_corrected_delta_carry(const __m256i *__restrict__ in,
                                            uint16_t *__restrict__ out,
                                            uint32_t bit,
                                            const __m256i *__restrict__ corrections,
                                            __m256i *__restrict__ carry,
                                            __m256i *__restrict__ sum);

// Sub-block variants for W=32/64/128 (2/4/8 OutRegs). Used by FoR-global
// with forWindowSize_=32/64/128. Same semantics as the 256-element versions.
void usimdpack_u16_n32(const uint16_t *__restrict__ in, __m256i *__restrict__ out, uint32_t bit);
void usimdpack_u16_n64(const uint16_t *__restrict__ in, __m256i *__restrict__ out, uint32_t bit);
void usimdpack_u16_n128(const uint16_t *__restrict__ in, __m256i *__restrict__ out, uint32_t bit);

void usimdunpack_u16_corrected_uniform_n32(const __m256i *__restrict__ in,
                                            uint16_t *__restrict__ out, uint32_t bit,
                                            __m256i anchor, __m256i *__restrict__ sum);
void usimdunpack_u16_corrected_uniform_n64(const __m256i *__restrict__ in,
                                            uint16_t *__restrict__ out, uint32_t bit,
                                            __m256i anchor, __m256i *__restrict__ sum);
void usimdunpack_u16_corrected_uniform_n128(const __m256i *__restrict__ in,
                                             uint16_t *__restrict__ out, uint32_t bit,
                                             __m256i anchor, __m256i *__restrict__ sum);

void usimdunpack_u16_corrected_n32(const __m256i *__restrict__ in,
                                    uint16_t *__restrict__ out, uint32_t bit,
                                    const __m256i *__restrict__ corrections,
                                    __m256i *__restrict__ sum);
void usimdunpack_u16_corrected_n64(const __m256i *__restrict__ in,
                                    uint16_t *__restrict__ out, uint32_t bit,
                                    const __m256i *__restrict__ corrections,
                                    __m256i *__restrict__ sum);
void usimdunpack_u16_corrected_n128(const __m256i *__restrict__ in,
                                     uint16_t *__restrict__ out, uint32_t bit,
                                     const __m256i *__restrict__ corrections,
                                     __m256i *__restrict__ sum);

} // namespace FastPForLib

#endif /* USIMDBITPACKING_U16_H_ */
