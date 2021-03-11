__global__ void create_newline_index(char *arr, long n, int *indices, long *result) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int offset = indices[index];

  long chars_per_thread = (n+stride-1) / stride;
  long start = index * chars_per_thread;
  long end = start + chars_per_thread;

  for (int i = start; i < end && i < n; i += 1) {
    if (arr[i] == '\n') {
      result[offset++] = i;
    }
  }
}
