/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Daniel Lemire
 */
#ifndef USIMDBITPACKING_NEW_H_
#define USIMDBITPACKING_NEW_H_

#include "common.h"

namespace FastPForLib {

void usimdunpack_new(const __m128i *__restrict__ in, uint32_t *__restrict__ out,
                 uint32_t bit, __m128i *__restrict__ sum_lo, __m128i *__restrict__ sum_hi);

} // namespace FastPForLib

#endif /* SIMDBITPACKING_H_ */
