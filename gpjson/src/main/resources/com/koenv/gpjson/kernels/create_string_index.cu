__global__ void create_string_index(long string_index_size, long *quote_index, char *quote_counts) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  long elements_per_thread = (string_index_size + stride - 1) / stride;
  long start = index * elements_per_thread;
  long end = start + elements_per_thread;

  // If the previous thread was in a string, we will inverse the string we get
  long previous_string = index > 0 && quote_counts[index - 1] == 1 ? 0xffffffffffffffffL : 0;

  for (long i = start; i < end && i < string_index_size; i += 1) {
    long quotes = quote_index[i];

    // https://github.com/simdjson/simdjson/blob/cfc965ff9ada688cf5950da829331b28dfcb949f/include/simdjson/arm64/bitmask.h
    quotes ^= quotes << 1;
    quotes ^= quotes << 2;
    quotes ^= quotes << 4;
    quotes ^= quotes << 8;
    quotes ^= quotes << 16;
    quotes ^= quotes << 32;

    // The prefix-XOR sum is combined with the previous string to negate it
    quotes = quotes ^ previous_string;

    quote_index[i] = quotes;

    // This will ensure that if we are currently in a string, the next block
    // will also be in a string.
    previous_string = quotes >> 63;
  }
}
