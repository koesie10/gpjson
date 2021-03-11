package com.koenv.gpjson.debug;

import com.koenv.gpjson.gpu.CUDARuntime;
import com.koenv.gpjson.gpu.ManagedGPUPointer;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;

public final class FormatUtils {
    private FormatUtils() {}

    public static void formatFileWithLongIndex(CUDARuntime cudaRuntime, ManagedGPUPointer file, ManagedGPUPointer index) {
        byte[] bytes = GPUUtils.readBytes(cudaRuntime, file);
        long[] indexData = GPUUtils.readLongs(cudaRuntime, index);

        formatFileWithLongIndex(bytes, indexData);
    }

    public static void formatFileWithLongIndex(byte[] file, long[] index) {
        byte[] emptyString = new byte[64];
        Arrays.fill(emptyString, (byte) ' ');
        byte[] escapeStringPart = new byte[64];
        for (int j = 0; j < 3; j++) {
            int start = j * 64;
            int end = Math.min(j * 64 + 64, file.length);

            System.arraycopy(emptyString, 0, escapeStringPart, 0, 64);
            System.arraycopy(file, start, escapeStringPart, 0, end - start);
            System.out.println(new String(escapeStringPart, StandardCharsets.UTF_8));
            System.out.println(new StringBuilder(String.format("%64s", Long.toBinaryString(index[j])).replace(' ', '0')).reverse().toString());
        }
    }
}
