#define MAX_NUM_LEVELS 3

__global__ void user_lang_query(char *file, long n, long *new_line_index, long new_line_index_size, long *string_index, long *leveled_bitmaps_index, long leveled_bitmaps_index_size, long level_size, long *result) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  long lines_per_thread = (new_line_index_size+stride-1) / stride;

  long start = index * lines_per_thread;
  long end = start + lines_per_thread;

  for (long i = start; i < end && i < new_line_index_size; i += 1) {
    long new_line_start = new_line_index[i];
    long new_line_end = (i + 1 < new_line_index_size) ? new_line_index[i+1] : n;

    int current_level = 0;
    int query_position = 0;
    long level_ends[MAX_NUM_LEVELS];
    level_ends[0] = new_line_end;
    for (int j = 1; j < MAX_NUM_LEVELS; j++) {
      level_ends[j] = -1;
    }

    // Property expression
    int current = 0;
    int looking_for_length = 4;
    char *looking_for = "user";

    for (long j = new_line_start; j < level_ends[current_level] && j < n; j += 1) {
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

        if (found) {
          current_level++;
          j++;

          // Now we need to find the end of the previous level (i.e. current_level - 1), unless we already have one
          if (level_ends[current_level] == -1) {
            for (long m = j + 1; m < level_ends[current_level - 1]; m += 1) {
              bool is_structural = (leveled_bitmaps_index[level_size * (current_level - 1) + m / 64] & (1L << m % 64)) != 0;
              if (is_structural) {
                level_ends[current_level] = m;
                break;
              }
            }
          }

          if (current == 0) {
            looking_for = "lang";
            current++;
          } else {
            // If we are storing a result, we are not in a string, so we can safely skip all whitespace
            // to find the start of the actual value
            char current_character;
            while (true) {
              current_character = file[j];
              if (current_character != '\n' && current_character != '\r' && current_character != '\t' && current_character != ' ') {
                  break;
              }

              j++;
            }

            int this_result_index = i*2;

            result[this_result_index] = j;
            result[this_result_index+1] = level_ends[current_level];

            goto end_single_line;
          }
        }
      }
    }

    end_single_line: ;
  }
}
