__global__ void create_combined_escape_newline_index(char *file, long n, bool *escape_carry_index, int *newline_count_index, long *escape_index, long escape_index_size, long *newline_index) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // We want to always calculate on 64-character boundaries, such that we can put
  // all bits of 64 characters into 1 long.
  long normal_chars_per_thread = (n+stride-1) / stride;
  long chars_per_thread = ((normal_chars_per_thread + 64 - 1) / 64) * 64;
  long start = index * chars_per_thread;
  long end = start + chars_per_thread;

  // Get the previous carry
  bool carry = index == 0 ? false : escape_carry_index[index - 1];

  // These are used for checking that not everything is escaped, because
  // we cannot deal with that scenario since that requires depending on all
  // previous carries, rather than just the previous carry.
  int escape_count = 0;
  int total_count = end - start;

  int final_loop_iteration = end;
  if (n < end) {
    final_loop_iteration = n;
  }

  int newline_offset = newline_count_index[index];

  for (long i = start; i < final_loop_iteration; i += 1) {
    char value = file[i];

    // If our last carry was 1, then we add it to the bit index here.
    // We do it here because we are actually setting the character that
    // is escaped, and not the escape character itself.
    if (carry == 1) {
      escape_index[i / 64] |= (1L << (i % 64));
    }

    if (value == '\\') {
      escape_count++;
      carry = carry ^ 1;
    } else {
      carry = 0;
    }

    if (value == '\n') {
      newline_index[newline_offset++] = i;
    }
  }

  // We do not expect to see a run of all backslashes
  assert(escape_count != total_count);
}
