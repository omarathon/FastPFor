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

} // namespace FastPForLib

#endif /* USIMDBITPACKING_U16_H_ */
