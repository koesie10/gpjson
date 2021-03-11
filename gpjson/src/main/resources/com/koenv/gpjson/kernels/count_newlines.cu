__global__ void count_newlines(char *arr, long n, int *result) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  long chars_per_thread = (n+stride-1) / stride;
  long start = index * chars_per_thread;
  long end = start + chars_per_thread;

  int count = 0;
  for (long i = start; i < end && i < n; i += 1) {
    if (arr[i] == '\n') {
      count += 1;
    }
  }

  result[index] = count;
}
