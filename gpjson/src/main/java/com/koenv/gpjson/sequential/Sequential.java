package com.koenv.gpjson.sequential;

import com.koenv.gpjson.debug.Timings;
import com.koenv.gpjson.jsonpath.ReadableIRByteBuffer;
import com.koenv.gpjson.stages.LeveledBitmapsIndex;

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

    public static long[] createLeveledBitmapsIndex(byte[] file, long[] stringIndex) {
        Timings.TIMINGS.start("Sequential#createLeveledBitmapsIndex");

        try {
            int levelSize = (file.length + 64 - 1) / 64;
            int numLevels = LeveledBitmapsIndex.NUM_LEVELS;
            int resultSize = levelSize * numLevels;

            long[] leveledBitmapsIndex = new long[resultSize];

            int level = -1;

            /*long[] bitIndex = new long[numLevels];*/

            for (int i = 0; i < file.length; i++) {
                assert (level >= -1 && level < numLevels);

                int offsetInBlock = i % 64;

                // Only if we're not in a string
                if ((stringIndex[i / 64] & (1L << offsetInBlock)) == 0) {
                    byte value = file[i];

                    if (value == '{' || value == '[') {
                        level++;
                    } else if (value == '}' || value == ']') {
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

    public static long[] findValue(byte[] file, long[] newlineIndex, long[] stringIndex, long[] leveledBitmapsIndex, ReadableIRByteBuffer query) {
        Timings.TIMINGS.start("Sequential#findValue");

        try {
            int levelSize = (file.length + 64 - 1) / 64;

            int currentLevel = 0;
            int lookingForType = query.readByte();

            int lookingForLength;
            byte[] lookingFor;

            switch (lookingForType) {
                case 0x01: // Dot expression
                    lookingForLength = query.readVarint();
                    lookingFor = query.readBytes(lookingForLength);
                    break;
                default:
                    throw new IllegalArgumentException("Invalid query type " + lookingForType);
            }

            long[] result = new long[newlineIndex.length];
            Arrays.fill(result, -1);

            for (int i = 0; i < newlineIndex.length; i++) {
                int newLineStart = (int) newlineIndex[i];
                int newLineEnd = (i + 1 < newlineIndex.length) ? (int) newlineIndex[i + 1] : file.length;

                new_line_loop:
                for (int j = newLineStart; j < newLineEnd && j < file.length; j += 1) {
                    boolean isStructural = (leveledBitmapsIndex[levelSize * currentLevel + j / 64] & (1L << j % 64)) != 0;

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

                        boolean found = true;

                        for (int k = 0; k < lookingForLength; k++) {
                            if (lookingFor[k] != file[string_start_index + k + 1]) {
                                found = false;
                                break;
                            }
                        }

                        if (!found) {
                            continue;
                        }

                        // This means we are at the correct key, so we'll increase our level
                        lookingForType = query.readByte();

                        switch (lookingForType) {
                            case 0x00:
                                result[i] = j;
                                break new_line_loop;
                            case 0x01: // Dot expression
                                currentLevel++;
                                lookingForLength = query.readVarint();
                                lookingFor = query.readBytes(lookingForLength);
                                break;
                            default:
                                throw new IllegalArgumentException("Invalid query type " + lookingForType);
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
