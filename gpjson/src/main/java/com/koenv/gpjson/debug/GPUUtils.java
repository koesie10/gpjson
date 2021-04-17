package com.koenv.gpjson.debug;

import com.koenv.gpjson.gpu.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public final class GPUUtils {
    private GPUUtils() {
    }

    public static byte[] readBytes(ManagedGPUPointer memory) {
        return readBytes(memory, -1);
    }

    public static byte[] readBytes(ManagedGPUPointer memory, int maxNumElements) {
        int numElements = (int) memory.size();
        if (maxNumElements > 0) {
            numElements = Math.min(numElements, maxNumElements);
        }

        ByteBuffer buffer = memory.copyToHost(numElements);

        byte[] result = new byte[numElements];
        int i = 0;
        while (buffer.hasRemaining()) {
            byte data = buffer.get();

            result[i++] = data;
        }

        return result;
    }

    public static int[] readInts(ManagedGPUPointer memory) {
        int numElements = (int) (memory.size() / Type.SINT32.getSizeBytes());

        ByteBuffer buffer = memory.copyToHost();

        int[] result = new int[numElements];
        int i = 0;
        while (buffer.hasRemaining()) {
            int data = buffer.getInt();

            result[i++] = data;
        }

        return result;
    }

    public static long[] readLongs(ManagedGPUPointer memory) {
        return readLongs(memory, -1);
    }

    public static long[] readLongs(ManagedGPUPointer memory, int maxNumElements) {
        int numElements = (int) (memory.size() / Type.SINT64.getSizeBytes());
        if (maxNumElements > 0) {
            numElements = Math.min(numElements, maxNumElements);
        }

        ByteBuffer buffer = memory.copyToHost(numElements * Type.SINT64.getSizeBytes());

        long[] result = new long[numElements];
        int i = 0;
        while (buffer.hasRemaining()) {
            long data = buffer.getLong();

            result[i++] = data;
        }

        return result;
    }
}
