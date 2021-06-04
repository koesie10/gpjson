// Should match with the value in LeveledBitmapsIndex
#define MAX_NUM_LEVELS 16

#define OPCODE_END 0x00
#define OPCODE_STORE_RESULT 0x01
#define OPCODE_MOVE_UP 0x02
#define OPCODE_MOVE_DOWN 0x03
#define OPCODE_MOVE_TO_KEY 0x04
#define OPCODE_MOVE_TO_INDEX 0x05
#define OPCODE_EXPRESSION_STRING_EQUALS 0x06

__global__ void find_value(char *file, long n, long *new_line_index, long new_line_index_size, long *string_index, long *leveled_bitmaps_index, long leveled_bitmaps_index_size, long level_size, char *query, int result_size, long *result) {
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

    int result_index = 0;

    char looking_for_type;

    // Property expression
    int looking_for_length;
    char *looking_for;

    // Index expression
    int looking_for_index;
    int current_index[MAX_NUM_LEVELS];

    for (int j = 0; j < MAX_NUM_LEVELS; j++) {
      current_index[j] = -1;
    }

    bool execute_ir = true;

    for (long j = new_line_start; j < level_ends[current_level] && j < n; j += 1) {
      while (execute_ir) {
        looking_for_type = query[query_position++];

        switch (looking_for_type) {
          case OPCODE_END: { // End
            goto end_single_line;
          }
          case OPCODE_STORE_RESULT: { // Store result
            assert(result_index < result_size);

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

            int this_result_index = i*2*result_size + result_index*2;

            result[this_result_index] = j;
            result[this_result_index+1] = level_ends[current_level];

            result_index++;

            break;
          }
          case OPCODE_MOVE_UP: { // Move up
            j = level_ends[current_level];

            level_ends[current_level] = -1;

            current_level--;
            break;
          }
          case OPCODE_MOVE_DOWN: { // Move down
            current_level++;

            // Now we need to find the end of the previous level (i.e. current_level - 1), unless we already have one
            if (level_ends[current_level] == -1) {
              for (long m = j + 1; m < level_ends[current_level - 1]; m += 1) {
                long current_level_index = leveled_bitmaps_index[level_size * (current_level - 1) + m / 64];
                if (current_level_index == 0) {
                  m += 64 - (m % 64) - 1;
                  continue;
                }
                bool is_structural = (current_level_index & (1L << m % 64)) != 0;
                if (is_structural) {
                  level_ends[current_level] = m;
                  break;
                }
              }
            }

            break;
          }
          case OPCODE_MOVE_TO_KEY: { // Move to key
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

            execute_ir = false;

            break;
          }
          case OPCODE_MOVE_TO_INDEX: { // Move to index
            looking_for_index = 0;
            int i = 0;
            int b;

            while (((b = query[query_position++]) & 0x80) != 0) {
              looking_for_index |= (b & 0x7F) << i;
              i += 7;
              assert(i <= 35);
            }
            looking_for_index = looking_for_index | (b << i);

            if (current_index[current_level] == -1) {
              char current_character;
              // Now find the opening `[`
              while (true) {
                current_character = file[j];
                if (current_character == '[') {
                  break;
                }
                assert(current_character == '\n' || current_character == '\r' || current_character == '\t' || current_character == ' ');

                j++;
                assert(j < n);
              }

              current_index[current_level] = 0;
            }

            if (looking_for_index == 0) {
              j++;

              execute_ir = true;
            } else {
              execute_ir = false;
            }

            break;
          }
          case OPCODE_EXPRESSION_STRING_EQUALS: {
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

            // Skip whitespace
            char current_character;
            while (true) {
              current_character = file[j];
              if (current_character != '\n' && current_character != '\r' && current_character != '\t' && current_character != ' ') {
                  break;
              }

              j++;
            }

            long actual_string_length = level_ends[current_level] - j;

            if (actual_string_length != looking_for_length) {
              goto end_single_line;
            }

            for (long k = 0; k < looking_for_length; k++) {
              if (looking_for[k] != file[j + k]) {
                goto end_single_line;
                break;
              }
            }

            break;
          }
          default: {
            assert(false);
            break;
          }
        }
      }

      long current_level_index = leveled_bitmaps_index[level_size * current_level + j / 64];
      if (current_level_index == 0) {
        j += 64 - (j % 64) - 1;
        continue;
      }
      bool is_structural = (current_level_index & (1L << j % 64)) != 0;

      if (looking_for_type == OPCODE_MOVE_TO_KEY) {
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

          execute_ir = true;

          for (long k = 0; k < looking_for_length; k++) {
            if (looking_for[k] != file[string_start_index + k + 1]) {
              execute_ir = false;
              break;
            }
          }
        }
      } else if (looking_for_type == OPCODE_MOVE_TO_INDEX) {
        if (is_structural && file[j] == ',') {
          current_index[current_level]++;
          if (looking_for_index == current_index[current_level]) {
            execute_ir = true;
          }
        }
      }
    }

    end_single_line: ;
  }
}
