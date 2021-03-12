__global__ void discover_structure(char *file, long n, long *string_index, char *structural_index) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // We want to always calculate on 64-character boundaries, to match what
  // we did in previous steps.
  long normal_chars_per_thread = (n+stride-1) / stride;
  long chars_per_thread = ((normal_chars_per_thread + 64 - 1) / 64) * 64;
  long start = index * chars_per_thread;
  long end = start + chars_per_thread;

  long strings = 0;

  for (long i = start; i < end && i < n; i += 1) {
    long offsetInBlock = i % 64;

    // At the start of each boundary (including the first), set the string characters
    if (offsetInBlock == 0) {
      strings = string_index[i / 64];
    }

    if ((strings & (1L << offsetInBlock)) != 0) {
      continue;
    }

    char value = file[i];

    if (value == 0x2c) {
      structural_index[i] = 0x01;
    } else if (value == 0x3a) {
      structural_index[i] = 0x02;
    } else if (value == 0x5b || value == 0x5d || value == 0x7b || value == 0x7d) {
      structural_index[i] = 0x04;
    } else if (value == 0x09 || value == 0x0a || value == 0x0d) {
      structural_index[i] = 0x08;
    } else if (value == 0x20) {
      structural_index[i] = 0x10;
    }
  }
}
