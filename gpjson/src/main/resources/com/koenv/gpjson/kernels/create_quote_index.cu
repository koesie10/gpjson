__global__ void create_quote_index(char *file, long n, long *escape_index, long *quote_index, char *quote_carry_index, long quote_index_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // We want to always calculate on 64-character boundaries, such that we can put
  // all bits of 64 characters into 1 long.
  long normal_chars_per_thread = (n+stride-1) / stride;
  long chars_per_thread = ((normal_chars_per_thread + 64 - 1) / 64) * 64;
  long start = index * chars_per_thread;
  long end = start + chars_per_thread;

  // This will contain the bitmask of escaped characters
  long escaped = 0;

  // Temporary variable for storing the current bit index
  long bit_index = 0;

  int quote_count = 0;

  for (long i = start; i < end && i < n; i += 1) {
    long offsetInBlock = i % 64;

    // At the start of each boundary (including the first), set the escaped characters
    if (offsetInBlock == 0) {
      escaped = escape_index[i / 64];
    }

    if (file[i] == '"') {
      if ((escaped & (1L << offsetInBlock)) == 0) {
        bit_index = bit_index | (1L << offsetInBlock);
        quote_count++;
      }
    }

    // If we are at the end of a boundary, set our result. We do not do it
    // if we are at the end since that would reset our bit_index.
    if (offsetInBlock == 63L && i != end - 1) {
      quote_index[i / 64] = bit_index;
      // Reset the bit index since we're starting over
      bit_index = 0;
    }
  }

  // In the final thread with data, we need to do this to make sure the last long is actually set
  int final_index = (end - 1) / 64;
  if (final_index < quote_index_size) {
    quote_index[final_index] = bit_index;
  }

  quote_carry_index[index] = quote_count & 1;
}
