package com.koenv.gpjson.debug;

import com.koenv.gpjson.gpu.ManagedGPUPointer;

import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

public final class FormatUtils {
    private FormatUtils() {}

    public static void formatFileWithLongIndex(ManagedGPUPointer file, ManagedGPUPointer index) {
        byte[] bytes = GPUUtils.readBytes(file);
        long[] indexData = GPUUtils.readLongs(index);

        formatFileWithLongIndex(bytes, indexData);
    }

    public static void formatFileWithLongIndex(byte[] file, long[] index) {
        byte[] emptyString = new byte[64];
        Arrays.fill(emptyString, (byte) ' ');
        byte[] escapeStringPart = new byte[64];
        for (int j = 0; j < (file.length + 64 - 1) / 64; j++) {
            int start = j * 64;
            int end = Math.min(j * 64 + 64, file.length);

            System.arraycopy(emptyString, 0, escapeStringPart, 0, 64);
            System.arraycopy(file, start, escapeStringPart, 0, end - start);
            System.out.println(new String(escapeStringPart, StandardCharsets.UTF_8).replace('\n', '↵'));
            System.out.println(new StringBuilder(String.format("%64s", Long.toBinaryString(index[j])).replace(' ', '0')).reverse().toString());
        }
    }

    public static void formatFileWithLongIndexToWriter(Writer w, byte[] file, long[] index) throws IOException {
        byte[] emptyString = new byte[64];
        Arrays.fill(emptyString, (byte) ' ');
        byte[] escapeStringPart = new byte[64];
        for (int j = 0; j < (file.length + 64 - 1) / 64; j++) {
            int start = j * 64;
            int end = Math.min(j * 64 + 64, file.length);

            System.arraycopy(emptyString, 0, escapeStringPart, 0, 64);
            System.arraycopy(file, start, escapeStringPart, 0, end - start);
            w.write(new String(escapeStringPart, StandardCharsets.UTF_8).replace('\n', '↵'));
            w.write('\n');
            w.write(new StringBuilder(String.format("%64s", Long.toBinaryString(index[j])).replace(' ', '0')).reverse().toString());
            w.write('\n');
        }
    }

    public static void formatFileWithByteIndex(ManagedGPUPointer file, ManagedGPUPointer index) {
        byte[] bytes = GPUUtils.readBytes(file);
        byte[] indexData = GPUUtils.readBytes(index);

        formatFileWithByteIndex(bytes, indexData);
    }

    public static void formatFileWithByteIndex(byte[] file, byte[] index) {
        StringBuilder stringBuilder = new StringBuilder();
        StringBuilder indexBuilder = new StringBuilder();
        for (int j = 0; j < 3; j++) {
            for (int i = 0; i < 64; i++) {
                int fileIndex = j * 64 + i;
                if (fileIndex < file.length) {
                    stringBuilder.append((char)file[fileIndex]);
                    indexBuilder.append(String.format("%02x", index[fileIndex]));
                } else {
                    stringBuilder.append(' ');
                    indexBuilder.append("  ");
                }

                indexBuilder.append(' ');
                stringBuilder.append("  ");
            }

            System.out.println(stringBuilder.toString());
            System.out.println(indexBuilder.toString());

            stringBuilder.setLength(0);
            indexBuilder.setLength(0);
        }
    }
}
