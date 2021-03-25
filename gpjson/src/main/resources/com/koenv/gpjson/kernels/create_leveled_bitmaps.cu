#define NUM_LEVELS 8

__global__ void create_leveled_bitmaps(char *file, long n, long *string_index, char *carry_index, long *structural_index, long structural_index_size, long level_size, char num_levels) {
  assert(num_levels == NUM_LEVELS);

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

  char level = carry_index[index];

  for (long i = start; i < end && i < n; i += 1) {
    long offsetInBlock = i % 64;

    // At the start of each boundary (including the first), set the string characters
    if (offsetInBlock == 0) {
      strings = string_index[i / 64];
    }

    // Do not process the character if we're in a string
    if ((strings & (1L << offsetInBlock)) != 0) {
      continue;
    }

    char value = file[i];

    if (value == '{' || value == '[') {
      level++;
    } else if (value == '}' || value == ']') {
      level--;
    } else if ((value == ':' || value == ',') && level >= 0 && level < NUM_LEVELS) {
      bit_index[level] = bit_index[level] | (1L << offsetInBlock);
    }

    // If we are at the end of a boundary, set our result. We do not do it
    // if we are at the end since that would reset our bit_index.
    if (offsetInBlock == 63L && i != end - 1) {
      for (int l = 0; l < NUM_LEVELS; l++) {
        structural_index[level_size * l + i / 64] = bit_index[l];
        // Reset the bit index since we're starting over
        bit_index[l] = 0;
      }
    }
  }

  // In the final thread with data, we need to do this to make sure the last longs are actually set
  int final_index = (end - 1) / 64;
  if (final_index < level_size) {
    for (int l = 0; l < NUM_LEVELS; l++) {
      structural_index[level_size * l + final_index] = bit_index[l];
    }
  }
}
