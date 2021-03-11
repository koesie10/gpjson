__global__ void discover_structure(char *arr, long n, char *result) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  long chars_per_thread = (n+stride-1) / stride;
  long start = index * chars_per_thread;
  long end = start + chars_per_thread;

  for (long i = start; i < end && i < n; i += 1) {
    if (arr[i] == '[') {
      result[i] = 0x01;
    } else if (arr[i] == '{') {
      result[i] = 0x02;
    } else if (arr[i] == ']') {
      result[i] = 0x04;
    } else if (arr[i] == '}') {
      result[i] = 0x08;
    } else if (arr[i] == ':') {
      result[i] = 0x10;
    } else if (arr[i] == ',') {
      result[i] = 0x20;
    }
  }
}
