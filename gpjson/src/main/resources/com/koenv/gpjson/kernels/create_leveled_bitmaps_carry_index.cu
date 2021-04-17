__global__ void create_leveled_bitmaps_carry_index(char *file, long n, long *string_index, char *level_carry_index) {
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

  signed char level = 0;

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
    }
  }

  level_carry_index[index] = level;
}
