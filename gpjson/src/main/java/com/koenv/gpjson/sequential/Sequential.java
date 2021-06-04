package com.koenv.gpjson.sequential;

import com.koenv.gpjson.debug.Timings;
import com.koenv.gpjson.jsonpath.IRByteInputBuffer;
import com.koenv.gpjson.jsonpath.JSONPathResult;
import com.koenv.gpjson.jsonpath.Opcode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Sequential {

    public static long[] createNewlineIndex(byte[] file) {
        Timings.TIMINGS.start("Sequential#createNewlineIndex");

        try {
            List<Long> index = new ArrayList<>();
            index.add(0L);

            for (int i = 0; i < file.length; i++) {
                if (file[i] != '\n') {
                    continue;
                }

                index.add((long) i);
            }

            return index.stream().mapToLong(value -> value).toArray();
        } finally {
            // Sequential#createNewlineIndex
            Timings.TIMINGS.end();
        }
    }

    public static long[] createStringIndex(byte[] file) {
        Timings.TIMINGS.start("Sequential#createStringIndex");

        try {
            long[] index = new long[(file.length + 64 - 1) / 64];
            byte escaped = 0;
            boolean inString = false;
            for (int i = 0; i < file.length; i++) {
                if (file[i] == '"' && escaped != 1) {
                    inString = !inString;
                }

                if (inString) {
                    index[i / 64] = index[i / 64] | (1L << (i % 64));
                }

                if (file[i] == '\\') {
                    escaped = (byte) (escaped ^ 1);
                } else {
                    escaped = 0;
                }
            }

            return index;
        } finally {
            // Sequential#createStringIndex
            Timings.TIMINGS.end();
        }
    }

    public static long[] createLeveledBitmapsIndex(byte[] file, long[] stringIndex, int numLevels) {
        Timings.TIMINGS.start("Sequential#createLeveledBitmapsIndex");

        try {
            int levelSize = (file.length + 64 - 1) / 64;
            int resultSize = levelSize * numLevels;

            long[] leveledBitmapsIndex = new long[resultSize];

            int level = -1;

            /*long[] bitIndex = new long[numLevels];*/

            for (int i = 0; i < file.length; i++) {
                assert (level >= -1);

                int offsetInBlock = i % 64;

                // Only if we're not in a string
                if ((stringIndex[i / 64] & (1L << offsetInBlock)) == 0) {
                    byte value = file[i];

                    if (value == '{' || value == '[') {
                        level++;
                        if (level < numLevels) {
                            leveledBitmapsIndex[levelSize * level + i / 64] |= (1L << offsetInBlock);
                        }
                    } else if (value == '}' || value == ']') {
                        if (level < numLevels) {
                            leveledBitmapsIndex[levelSize * level + i / 64] |= (1L << offsetInBlock);
                        }
                        level--;
                    } else if ((value == ':' || value == ',') && level >= 0 && level < numLevels) {
                        leveledBitmapsIndex[levelSize * level + i / 64] |= (1L << offsetInBlock);
                        /*bitIndex[level] |= (1L << offsetInBlock);*/
                    }
                }

            /*// If we are at the end of a boundary, set our result. We do not do it
            // if we are at the end since that would reset our bit_index.
            if (offsetInBlock == 63L && i != file.length - 1) {
                for (int l = 0; l < numLevels; l++) {
                    leveledBitmapsIndex[levelSize * l + i / 64] = bitIndex[l];
                    // Reset the bit index since we're starting over
                    bitIndex[l] = 0;
                }
            }*/
            }

        /*// In the final thread with data, we need to do this to make sure the last longs are actually set
        int final_index = (file.length - 1) / 64;
        if (final_index < levelSize) {
            for (int l = 0; l < numLevels; l++) {
                leveledBitmapsIndex[levelSize * l + final_index] = bitIndex[l];
            }
        }*/

            return leveledBitmapsIndex;
        } finally {
            // Sequential#createLeveledBitmapsIndex
            Timings.TIMINGS.end();
        }
    }

    public static long[] findValue(byte[] file, long[] newlineIndex, long[] stringIndex, long[] leveledBitmapsIndex, JSONPathResult query) {
        int MAX_NUM_LEVELS = 16;

        Timings.TIMINGS.start("Sequential#findValue");

        IRByteInputBuffer queryBuffer = query.getIr().toReadable();
        queryBuffer.mark();

        int resultSize = query.getNumResults();

        try {
            int levelSize = (file.length + 64 - 1) / 64;

            long[] result = new long[newlineIndex.length * 2 * resultSize];
            Arrays.fill(result, -1);

            for (int i = 0; i < newlineIndex.length; i++) {
                int newLineStart = (int) newlineIndex[i];
                int newLineEnd = (i + 1 < newlineIndex.length) ? (int) newlineIndex[i + 1] : file.length;

                queryBuffer.reset();
                queryBuffer.mark();

                int currentLevel = 0;
                int[] levelEnds = new int[MAX_NUM_LEVELS];
                levelEnds[0] = newLineEnd;
                for (int j = 1; j < MAX_NUM_LEVELS; j++) {
                    levelEnds[j] = -1;
                }

                int resultIndex = 0;

                Opcode lookingForType = null;

                // Property expression
                int lookingForLength = -1;
                byte[] lookingFor = null;

                // Index expression
                int lookingForIndex = -1;
                int[] currentIndex = new int[MAX_NUM_LEVELS];

                for (int j = 0; j < MAX_NUM_LEVELS; j++) {
                    currentIndex[j] = -1;
                }

                boolean executeIr = true;

                new_line_loop:
                for (int j = newLineStart; j < levelEnds[currentLevel] && j < file.length; j += 1) {
                    while (executeIr) {
                        lookingForType = queryBuffer.readOpcode();

                        switch (lookingForType) {
                            case END:
                                break new_line_loop;
                            case STORE_RESULT:
                                assert (resultIndex < resultSize);

                                // If we are storing a result, we are not in a string, so we can safely skip all whitespace
                                // to find the start of the actual value
                                while (true) {
                                    byte currentCharacter = file[j];
                                    if (currentCharacter != '\n' && currentCharacter != '\r' && currentCharacter != '\t' && currentCharacter != ' ') {
                                        break;
                                    }

                                    j++;
                                }

                                int thisResultIndex = i * 2 * resultSize + resultIndex * 2;

                                result[thisResultIndex] = j;
                                result[thisResultIndex + 1] = levelEnds[currentLevel];

                                resultIndex++;

                                break;
                            case MOVE_UP:
                                j = levelEnds[currentLevel];

                                levelEnds[currentLevel] = -1;

                                currentLevel--;

                                break;
                            case MOVE_DOWN:
                                currentLevel++;

                                // Now we need to find the end of the previous level (i.e. current_level - 1), unless we already have one
                                if (levelEnds[currentLevel] == -1) {
                                    for (int m = j + 1; m < levelEnds[currentLevel - 1]; m += 1) {
                                        boolean isStructural = (leveledBitmapsIndex[levelSize * (currentLevel - 1) + m / 64] & (1L << m % 64)) != 0;
                                        if (isStructural) {
                                            levelEnds[currentLevel] = m;
                                            break;
                                        }
                                    }
                                }

                                break;
                            case MOVE_TO_KEY:
                                lookingForLength = queryBuffer.readVarint();
                                lookingFor = queryBuffer.readBytes(lookingForLength);

                                executeIr = false;

                                break;
                            case MOVE_TO_INDEX:
                                lookingForIndex = queryBuffer.readVarint();

                                if (currentIndex[currentLevel] == -1) {
                                    byte currentCharacter;
                                    // Now find the opening `[`
                                    while (true) {
                                        currentCharacter = file[j];
                                        if (currentCharacter == '[') {
                                            break;
                                        }
                                        assert (currentCharacter == '\n' || currentCharacter == '\r' || currentCharacter == '\t' || currentCharacter == ' ');

                                        j++;
                                        assert (j < file.length);
                                    }

                                    currentIndex[currentLevel] = 0;
                                }

                                if (lookingForIndex == 0) {
                                    j++;

                                    executeIr = true;
                                } else {
                                    executeIr = false;
                                }

                                break;
                            case EXPRESSION_STRING_EQUALS:
                                lookingForLength = queryBuffer.readVarint();
                                lookingFor = queryBuffer.readBytes(lookingForLength);

                                // Skip whitespace
                                while (true) {
                                    byte currentCharacter = file[j];
                                    if (currentCharacter != '\n' && currentCharacter != '\r' && currentCharacter != '\t' && currentCharacter != ' ') {
                                        break;
                                    }

                                    j++;
                                }

                                int actualStringLength = levelEnds[currentLevel] - j;

                                if (actualStringLength != lookingForLength) {
                                    break new_line_loop;
                                }

                                for (int k = 0; k < lookingForLength; k++) {
                                    if (lookingFor[k] != file[j + k]) {
                                        break new_line_loop;
                                    }
                                }

                                break;
                            default:
                                throw new IllegalStateException("Illegal opcode " + lookingForType);
                        }
                    }

                    boolean isStructural = (leveledBitmapsIndex[levelSize * currentLevel + j / 64] & (1L << j % 64)) != 0;

                    if (lookingForType == Opcode.MOVE_TO_KEY) {
                        if (isStructural && file[j] == ':') {
                            // Start looking for the end of the string
                            int string_end_index = -1;
                            for (int k = j - 1; k > newLineStart; k -= 1) {
                                if ((stringIndex[k / 64] & (1L << k % 64)) != 0) {
                                    string_end_index = k;
                                    break;
                                }
                            }

                            assert (string_end_index >= newLineStart);

                            int string_start_index = string_end_index - lookingForLength;
                            if (string_start_index < newLineStart || file[string_start_index] != '"') {
                                continue;
                            }

                            executeIr = true;

                            for (int k = 0; k < lookingForLength; k++) {
                                if (lookingFor[k] != file[string_start_index + k + 1]) {
                                    executeIr = false;
                                    break;
                                }
                            }
                        }
                    } else if (lookingForType == Opcode.MOVE_TO_INDEX) {
                        if (isStructural && file[j] == ',') {
                            currentIndex[currentLevel]++;
                            if (lookingForIndex == currentIndex[currentLevel]) {
                                executeIr = true;
                            }
                        }
                    }
                }
            }

            return result;
        } finally {
            // Sequential#findValue
            Timings.TIMINGS.end();
        }
    }
}
