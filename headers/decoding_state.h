#ifndef DECODING_STATE_H_
#define DECODING_STATE_H_

#include "common.h"

extern size_t g_decoding_idx;
extern size_t g_next_exception_idx;
extern int32_t g_delta_sum;

extern const uint32_t *__restrict__ g_i;
extern const uint32_t *__restrict__ g_end_exception;

#endif /* DECODING_STATE_H_ */
