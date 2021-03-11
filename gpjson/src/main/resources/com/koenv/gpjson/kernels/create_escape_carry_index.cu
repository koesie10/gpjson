__global__ void create_escape_carry_index(char *file, long n, char *escape_carry_index) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  long chars_per_thread = max((n+stride-1) / stride, 64L);
  long start = index * chars_per_thread;
  long end = start + chars_per_thread;

  // There are essentially two cases:
  // - The last character in the previous block is an escape character.
  // - The last character in the previous block is not an escape character.
  // However, we don't know in advance which one it is, because
  // we are not sequential. So, here we'll basically
  // calculate the carry of each thread assuming the initial
  // carry is 0.

  long carry = 0;

  for (long i = start; i < end && i < n; i += 1) {
    if (file[i] == '\\') {
      carry = 1 ^ carry;
    } else {
      carry = 0;
    }
  }

  escape_carry_index[index] = carry;
}
