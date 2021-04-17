#define NUM_LEVELS 22

__global__ void create_leveled_bitmaps(char *file, long n, long *string_index, char *carry_index, long *leveled_bitmaps_index, long leveled_bitmaps_index_size, long level_size, int num_levels) {
  assert(num_levels <= NUM_LEVELS);

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // We want to always calculate on 64-character boundaries, to match what
  // we did in previous steps.
  long normal_chars_per_thread = (n+stride-1) / stride;
  long chars_per_thread = ((normal_chars_per_thread + 64 - 1) / 64) * 64;
  long start = index * chars_per_thread;
  long end = start + chars_per_thread;

  // Temporary variable for storing the current string index
  long strings = 0;

  // Temporary variable for storing the current bit indexes
  long bit_index[NUM_LEVELS];

  // Initialize bit index to 0
  for (int l = 0; l < NUM_LEVELS; l++) {
    bit_index[l] = 0;
  }

  signed char level = carry_index[index];

  int final_loop_iteration = end;
  if (n < end) {
    final_loop_iteration = n;
  }

  for (long i = start; i < final_loop_iteration; i += 1) {
    assert(level >= -1);

    long offsetInBlock = i % 64;

    // At the start of each boundary (including the first), set the string characters
    if (offsetInBlock == 0) {
      strings = string_index[i / 64];
    }

    // Only if we're not in a string
    if ((strings & (1L << offsetInBlock)) == 0) {
      char value = file[i];

      if (value == '{' || value == '[') {
        level++;
      } else if (value == '}' || value == ']') {
        level--;
      } else if ((value == ':' || value == ',') && level >= 0 && level < num_levels) {
        bit_index[level] = bit_index[level] | (1L << offsetInBlock);
      }
    }

    // If we are at the end of a boundary, set our result.
    if (offsetInBlock == 63L) {
      for (int l = 0; l < num_levels; l++) {
        leveled_bitmaps_index[level_size * l + i / 64] = bit_index[l];
        // Reset the bit index since we're starting over
        bit_index[l] = 0;
      }
    }
  }

  if (n < end && (final_loop_iteration - 1) % 64 != 63L && n - start > 0) {
    // In the final thread with data, we need to do this to make sure the last longs are actually set
    int final_index = (final_loop_iteration - 1) / 64;
    if (final_index < level_size) {
      for (int l = 0; l < num_levels; l++) {
        leveled_bitmaps_index[level_size * l + final_index] = bit_index[l];
      }
    }
  }
}
