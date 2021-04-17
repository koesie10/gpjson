package com.koenv.gpjson.stages;

import com.koenv.gpjson.KernelTest;
import com.koenv.gpjson.debug.GPUUtils;
import com.koenv.gpjson.gpu.ManagedGPUPointer;
import com.koenv.gpjson.gpu.Type;
import com.koenv.gpjson.sequential.Sequential;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class LeveledBitmapsIndexTest extends KernelTest {
    @ParameterizedTest
    @ValueSource(strings = {
            "simple",
            "boundary",
    })
    public void create(String name) throws IOException {
        try (
                ManagedGPUPointer fileMemory = readFileToGPU("stages/leveled_bitmaps_index/" + name + ".json");
                ManagedGPUPointer stringIndexMemory = cudaRuntime.allocateUnmanagedMemory((fileMemory.size() + 64 - 1) / 64, Type.SINT64);
        ) {
            ByteBuffer stringIndex = createStringIndex(name);
            stringIndexMemory.loadFrom(stringIndex);

            LeveledBitmapsIndex leveledBitmapsIndex = new LeveledBitmapsIndex(cudaRuntime, fileMemory, stringIndexMemory);
            try (ManagedGPUPointer indexMemory = leveledBitmapsIndex.create()) {
                long[] leveledBitmaps = GPUUtils.readLongs(indexMemory);
                long[] expectedLeveledBitmaps = createExpectedLeveledBitmapsIndex(name);

                assertArrayEquals(expectedLeveledBitmaps, leveledBitmaps);
            }
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {
            "simple",
            "boundary",
    })
    public void leveledBitmapIndexSteps(String name) throws IOException {
        try (
                ManagedGPUPointer fileMemory = readFileToGPU("stages/leveled_bitmaps_index/" + name + ".json");
                ManagedGPUPointer stringIndexMemory = cudaRuntime.allocateUnmanagedMemory((fileMemory.size() + 64 - 1) / 64, Type.SINT64);
                ManagedGPUPointer carryIndexMemory = cudaRuntime.allocateUnmanagedMemory(LeveledBitmapsIndex.CARRY_INDEX_SIZE, Type.SINT8);
                ManagedGPUPointer leveledBitmapsIndexMemory = cudaRuntime.allocateUnmanagedMemory(((fileMemory.size() + 64 - 1) / 64) * LeveledBitmapsIndex.NUM_LEVELS, Type.SINT64);
        ) {
            stringIndexMemory.loadFrom(createStringIndex(name));

            LeveledBitmapsIndex leveledBitmapsIndex = new LeveledBitmapsIndex(cudaRuntime, fileMemory, stringIndexMemory);

            leveledBitmapsIndex.createCarryIndex(carryIndexMemory);

            byte[] carries = GPUUtils.readBytes(carryIndexMemory);
            byte[] expectedCarries = createExpectedLeveledBitmapsCarryIndex(name);

            assertArrayEquals(expectedCarries, carries);

            leveledBitmapsIndex.createLevelsIndex(carryIndexMemory);

            byte[] levelCarries = GPUUtils.readBytes(carryIndexMemory);
            byte[] expectedLevelCarries = createExpectedLeveledBitmapsLevelCarryIndex(name);

            assertArrayEquals(expectedLevelCarries, levelCarries);

            leveledBitmapsIndex.createLeveledBitmaps(leveledBitmapsIndexMemory, carryIndexMemory);

            long[] leveledBitmaps = GPUUtils.readLongs(leveledBitmapsIndexMemory);
            long[] expectedLeveledBitmaps = createExpectedLeveledBitmapsIndex(name);

            assertArrayEquals(expectedLeveledBitmaps, leveledBitmaps);
        }
    }

    private ByteBuffer createStringIndex(String name) throws IOException {
        byte[] bytes = readFile("stages/leveled_bitmaps_index/" + name + ".json");

        long[] index = Sequential.createStringIndex(bytes);

        ByteBuffer buffer = ByteBuffer.allocateDirect(index.length * 8).order(ByteOrder.LITTLE_ENDIAN);
        for (long l : index) {
            buffer.putLong(l);
        }

        return buffer;
    }

    private byte[] createExpectedLeveledBitmapsCarryIndex(String name) throws IOException {
        byte[] bytes = readFile("stages/leveled_bitmaps_index/" + name + ".json");

        long[] stringIndex = Sequential.createStringIndex(bytes);

        int normalCharsPerThread = (bytes.length + LeveledBitmapsIndex.CARRY_INDEX_SIZE - 1) / LeveledBitmapsIndex.CARRY_INDEX_SIZE;
        int charsPerThread = ((normalCharsPerThread + 64 - 1) / 64) * 64;

        byte[] index = new byte[LeveledBitmapsIndex.CARRY_INDEX_SIZE];

        for (int i = 0; i < bytes.length; i++) {
            int offsetInBlock = i % 64;

            // Only if we're not in a string
            if ((stringIndex[i / 64] & (1L << offsetInBlock)) == 0) {
                byte value = bytes[i];

                if (value == '{' || value == '[') {
                    index[i / charsPerThread]++;
                } else if (value == '}' || value == ']') {
                    index[i / charsPerThread]--;
                }
            }
        }

        return index;
    }

    private byte[] createExpectedLeveledBitmapsLevelCarryIndex(String name) throws IOException {
        byte[] bytes = readFile("stages/leveled_bitmaps_index/" + name + ".json");

        long[] stringIndex = Sequential.createStringIndex(bytes);

        int normalCharsPerThread = (bytes.length + LeveledBitmapsIndex.CARRY_INDEX_SIZE - 1) / LeveledBitmapsIndex.CARRY_INDEX_SIZE;
        int charsPerThread = ((normalCharsPerThread + 64 - 1) / 64) * 64;

        byte[] index = new byte[LeveledBitmapsIndex.CARRY_INDEX_SIZE];
        Arrays.fill(index, (byte) -1);

        byte level = -1;

        for (int i = 0; i < bytes.length; i++) {
            int offsetInBlock = i % 64;

            // Only if we're not in a string
            if ((stringIndex[i / 64] & (1L << offsetInBlock)) == 0) {
                byte value = bytes[i];

                if (value == '{' || value == '[') {
                    level++;
                } else if (value == '}' || value == ']') {
                    level--;
                }
            }

            if (i / charsPerThread + 1 < index.length) {
                index[i / charsPerThread + 1] = level;
            }
        }

        return index;
    }

    private long[] createExpectedLeveledBitmapsIndex(String name) throws IOException {
        byte[] bytes = readFile("stages/leveled_bitmaps_index/" + name + ".json");

        long[] stringIndex = Sequential.createStringIndex(bytes);

        return Sequential.createLeveledBitmapsIndex(bytes, stringIndex);
    }
}
