#ifndef _DECODING_STATE_H
#define _DECODING_STATE_H

#include "common.h"

#include <array>

extern std::array<__m128i, 32> g_delta_sum_masks;
extern size_t g_decode_counter;

#endif