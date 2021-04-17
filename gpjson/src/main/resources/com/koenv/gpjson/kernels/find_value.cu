__global__ void find_value(char *file, long n, long *new_line_index, long new_line_index_size, long *string_index, long *leveled_bitmaps_index, long leveled_bitmaps_index_size, long level_size, int num_levels, long *result) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  long lines_per_thread = (new_line_index_size+stride-1) / stride;

  long start = index * lines_per_thread;
  long end = start + lines_per_thread;

  int current_level = 0;
  char *looking_for = "user";
  int looking_for_length = 4;

  for (long i = start; i < end && i < new_line_index_size; i += 1) {
    result[i] = -1;
    long new_line_start = new_line_index[i];
    long new_line_end = (i + 1 < new_line_index_size) ? new_line_index[i+1] : n;

    for (long j = new_line_start; j < new_line_end && j < n; j += 1) {
      bool is_structural = (leveled_bitmaps_index[level_size * current_level + j / 64] & (1L << j % 64)) != 0;

      if (is_structural && file[j] == ':') {
        // Start looking for the end of the string
        long string_end_index = -1;
        for (long k = j - 1; k > new_line_start; k -= 1) {
          if ((string_index[k / 64] & (1L << k % 64)) != 0) {
            string_end_index = k;
            break;
          }
        }

        assert(string_end_index >= new_line_start);

        long string_start_index = string_end_index - looking_for_length;
        if (string_start_index < new_line_start || file[string_start_index] != '"') {
          continue;
        }

        bool found = true;

        for (long k = 0; k < looking_for_length; k++) {
          if (looking_for[k] != file[string_start_index + k + 1]) {
            found = false;
            break;
          }
        }

        if (!found) {
          continue;
        }

        // This means we are at the correct key, so we'll increase our level
        if (current_level == 0) {
          current_level++;
          looking_for = "lang";
        } else if (current_level == 1) {
          result[i] = j;
          break;
        }
      }
    }
  }
}
