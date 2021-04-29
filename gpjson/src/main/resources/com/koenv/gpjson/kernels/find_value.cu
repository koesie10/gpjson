__global__ void find_value(char *file, long n, long *new_line_index, long new_line_index_size, long *string_index, long *leveled_bitmaps_index, long leveled_bitmaps_index_size, long level_size, char *query, int query_size, long *result) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  long lines_per_thread = (new_line_index_size+stride-1) / stride;

  long start = index * lines_per_thread;
  long end = start + lines_per_thread;

  for (long i = start; i < end && i < new_line_index_size; i += 1) {
    result[i] = -1;
    long new_line_start = new_line_index[i];
    long new_line_end = (i + 1 < new_line_index_size) ? new_line_index[i+1] : n;

    int current_level = 0;
    int query_position = 0;
    long current_level_end = new_line_end;

    char looking_for_type = query[query_position++];

    int looking_for_length;
    char *looking_for;

    int looking_for_index;
    int current_index;

    switch (looking_for_type) {
      case 0x01: {// Property expression
        looking_for_length = 0;
        int i = 0;
        int b;

        while (((b = query[query_position++]) & 0x80) != 0) {
          looking_for_length |= (b & 0x7F) << i;
          i += 7;
          assert(i <= 35);
        }
        looking_for_length = looking_for_length | (b << i);

        looking_for = query + query_position;
        query_position += looking_for_length;

        break;
      }
      default: {
        assert(false);
        break;
      }
    }

    for (long j = new_line_start; j < current_level_end && j < n; j += 1) {
      bool is_structural = (leveled_bitmaps_index[level_size * current_level + j / 64] & (1L << j % 64)) != 0;

      bool move_to_next_level = false;

      if (looking_for_type == 0x01) {
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

          move_to_next_level = true;

          for (long k = 0; k < looking_for_length; k++) {
            if (looking_for[k] != file[string_start_index + k + 1]) {
              move_to_next_level = false;
              break;
            }
          }
        }
      } else if (looking_for_type == 0x02) {
        if (looking_for_index == 0) {
          // This will only happen once
          move_to_next_level = true;
        } else if (is_structural && file[j] == ',') {
           current_index++;
           if (looking_for_index == current_index) {
             move_to_next_level = true;
           }
        }
      }

      if (move_to_next_level) {
        looking_for_type = query[query_position++];

        switch (looking_for_type) {
          case 0x00: { // End
            result[i] = j;
            goto end_single_line;
          }
          case 0x01: { // Property expression
            looking_for_length = 0;
            int i = 0;
            int b;

            while (((b = query[query_position++]) & 0x80) != 0) {
              looking_for_length |= (b & 0x7F) << i;
              i += 7;
              assert(i <= 35);
            }
            looking_for_length = looking_for_length | (b << i);

            looking_for = query + query_position;
            query_position += looking_for_length;

            current_level++;

            break;
          }
          case 0x02: { // Index expression
            looking_for_index = 0;
            int i = 0;
            int b;

            while (((b = query[query_position++]) & 0x80) != 0) {
              looking_for_index |= (b & 0x7F) << i;
              i += 7;
              assert(i <= 35);
            }
            looking_for_index = looking_for_index | (b << i);

            current_level++;
            current_index = 0;

            char current_character = file[j];
            // Now find the opening `[`
            while (true) {
              j++;
              assert(j < n);
              current_character = file[j];
              if (current_character == '[') {
                break;
              }
              assert(current_character == '\n' || current_character == '\r' || current_character == '\t' || current_character == ' ');
            }

            break;
          }
          default: {
            assert(false);
            break;
          }
        }

        // Now we need to find the end of the previous level (i.e. current_level - 1)
        for (long m = j + 1; m < current_level_end; m += 1) {
          bool is_structural = (leveled_bitmaps_index[level_size * (current_level - 1) + m / 64] & (1L << m % 64)) != 0;
          if (is_structural) {
            current_level_end = m;
            break;
          }
        }
      }
    }

    end_single_line: ;
  }
}
