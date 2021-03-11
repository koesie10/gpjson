package com.koenv.gpjson.debug;

import com.koenv.gpjson.gpu.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public final class GPUUtils {
    private GPUUtils() {
    }

    public static byte[] readBytes(CUDARuntime cudaRuntime, ManagedGPUPointer memory) {
        int size = (int) memory.size();

        ByteBuffer buffer = memory.copyToHost();

        byte[] result = new byte[size];
        int i = 0;
        while (buffer.hasRemaining()) {
            byte data = buffer.get();

            result[i++] = data;
        }

        return result;
    }

    public static int[] readInts(CUDARuntime cudaRuntime, ManagedGPUPointer memory) {
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

    public static long[] readLongs(CUDARuntime cudaRuntime, ManagedGPUPointer memory) {
        int numElements = (int) (memory.size() / Type.SINT64.getSizeBytes());

        ByteBuffer buffer = memory.copyToHost();

        long[] result = new long[numElements];
        int i = 0;
        while (buffer.hasRemaining()) {
            long data = buffer.getLong();

            result[i++] = data;
        }

        return result;
    }
}
