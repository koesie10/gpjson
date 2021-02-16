__global__ void discover_structure(char *arr, long n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (arr[idx] == ',') {
    arr[idx] = 1;
  } else {
    arr[idx] = 0;
  }
}
