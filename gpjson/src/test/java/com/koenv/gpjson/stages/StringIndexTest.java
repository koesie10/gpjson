package com.koenv.gpjson.stages;

import com.koenv.gpjson.KernelTest;
import com.koenv.gpjson.debug.GPUUtils;
import com.koenv.gpjson.gpu.ManagedGPUPointer;
import com.koenv.gpjson.gpu.Type;
import com.koenv.gpjson.sequential.Sequential;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.IOException;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class StringIndexTest extends KernelTest {
    @ParameterizedTest
    @ValueSource(strings = {
            "simple",
            "escape_carry",
            "very_long_string",
    })
    public void create(String name) throws IOException {
        try (
                ManagedGPUPointer fileMemory = readFileToGPU("stages/string_index/" + name + ".txt")
        ) {
            StringIndex stringIndex = new StringIndex(cudaRuntime, fileMemory);
            try (ManagedGPUPointer indexMemory = stringIndex.create()) {
                long[] strings = GPUUtils.readLongs(indexMemory);
                long[] expectedStrings = createExpectedStringIndex(name);

                assertArrayEquals(expectedStrings, strings);
            }
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {
            "simple",
            "escape_carry",
            "very_long_string"
    })
    public void stringIndexSteps(String name) throws IOException {
        try (
                ManagedGPUPointer fileMemory = readFileToGPU("stages/string_index/" + name + ".txt");
                ManagedGPUPointer carryIndexMemory = cudaRuntime.allocateUnmanagedMemory(StringIndex.CARRY_INDEX_SIZE);
                ManagedGPUPointer escapeIndexMemory = cudaRuntime.allocateUnmanagedMemory((fileMemory.size() + 64 - 1) / 64, Type.SINT64);
                ManagedGPUPointer stringIndexMemory = cudaRuntime.allocateUnmanagedMemory((fileMemory.size() + 64 - 1) / 64, Type.SINT64)
        ) {
            StringIndex stringIndex = new StringIndex(cudaRuntime, fileMemory);

            stringIndex.createEscapeCarryIndex(carryIndexMemory);

            byte[] escapeCarries = GPUUtils.readBytes(carryIndexMemory);
            byte[] expectedEscapeCarries = createExpectedEscapeCarryIndex(name);

            assertArrayEquals(expectedEscapeCarries, escapeCarries);

            stringIndex.createEscapeIndex(carryIndexMemory, escapeIndexMemory);

            long[] escapes = GPUUtils.readLongs(escapeIndexMemory);
            long[] expectedEscapes = createExpectedEscapeIndex(name);

            assertArrayEquals(expectedEscapes, escapes);

            stringIndex.createQuoteIndex(escapeIndexMemory, stringIndexMemory, carryIndexMemory);

            long[] quotes = GPUUtils.readLongs(stringIndexMemory);
            long[] expectedQuotes = createExpectedQuoteIndex(name);

            assertArrayEquals(expectedQuotes, quotes);

            byte[] quoteCarries = GPUUtils.readBytes(carryIndexMemory);
            byte[] expectedQuoteCarries = createExpectedQuoteCarryIndex(quotes);

            assertArrayEquals(expectedQuoteCarries, quoteCarries);

            stringIndex.createStringIndex(stringIndexMemory, carryIndexMemory);

            long[] strings = GPUUtils.readLongs(stringIndexMemory);
            long[] expectedStrings = createExpectedStringIndex(name);

            assertArrayEquals(expectedStrings, strings);

            // FormatUtils.formatFileWithLongIndex(cudaRuntime, fileMemory, stringIndexMemory);
        }
    }

    private byte[] createExpectedEscapeCarryIndex(String name) throws IOException {
        byte[] bytes = readFile("stages/string_index/" + name + ".txt");

        int normalCharactersPerThread = Math.max((bytes.length + StringIndex.CARRY_INDEX_SIZE - 1) / StringIndex.CARRY_INDEX_SIZE, 64);
        int charactersPerThread = ((normalCharactersPerThread + 64 - 1) / 64) * 64;

        byte[] index = new byte[StringIndex.CARRY_INDEX_SIZE];

        for (int i = 0; i < bytes.length; i++) {
            int newCarry = 0;
            if (bytes[i] == '\\') {
                newCarry = 1 ^ index[i / charactersPerThread];
            }

            index[i / charactersPerThread] = (byte) newCarry;
        }

        return index;
    }

    private long[] createExpectedEscapeIndex(String name) throws IOException {
        byte[] bytes = readFile("stages/string_index/" + name + ".txt");

        long[] index = new long[(bytes.length + 64 - 1) / 64];
        byte carry = 0;
        for (int i = 0; i < bytes.length; i++) {
            if (carry == 1) {
                index[i / 64] = index[i / 64] | (1L << (i % 64));
            }

            if (bytes[i] == '\\') {
                carry = (byte) (carry ^ 1);
            } else {
                carry = 0;
            }
        }

        return index;
    }

    private long[] createExpectedQuoteIndex(String name) throws IOException {
        byte[] bytes = readFile("stages/string_index/" + name + ".txt");

        long[] index = new long[(bytes.length + 64 - 1) / 64];
        byte carry = 0;
        for (int i = 0; i < bytes.length; i++) {
            if (bytes[i] == '"' && carry != 1) {
                index[i / 64] = index[i / 64] | (1L << (i % 64));
            }

            if (bytes[i] == '\\') {
                carry = (byte) (carry ^ 1);
            } else {
                carry = 0;
            }
        }

        return index;
    }

    private byte[] createExpectedQuoteCarryIndex(long[] quotes) {
        byte[] index = new byte[StringIndex.CARRY_INDEX_SIZE];

        int quotesPerThread = (quotes.length + StringIndex.CARRY_INDEX_SIZE - 1) / StringIndex.CARRY_INDEX_SIZE;

        int quoteCount = 0;

        for (int i = 0; i < StringIndex.CARRY_INDEX_SIZE; i++) {
            int start = i * quotesPerThread;
            int end = start + quotesPerThread;

            for (int j = start; j < end && j < quotes.length; j++) {
                quoteCount += Long.bitCount(quotes[j]);
            }

            index[i] = (byte) (quoteCount & 1);
        }

        return index;
    }

    private long[] createExpectedStringIndex(String name) throws IOException {
        byte[] bytes = readFile("stages/string_index/" + name + ".txt");

        return Sequential.createStringIndex(bytes);
    }
}
