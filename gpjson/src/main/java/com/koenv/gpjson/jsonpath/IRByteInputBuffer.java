package com.koenv.gpjson.jsonpath;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

public class IRByteInputBuffer {
    private final ByteBuffer buffer;

    public IRByteInputBuffer(ByteBuffer buffer) {
        this.buffer = buffer;
    }

    public byte readByte() {
        return buffer.get();
    }

    public byte[] readBytes(int length) {
        byte[] result = new byte[length];
        buffer.get(result);
        return result;
    }

    public int readVarint() {
        int value = 0;
        int i = 0;
        int b;
        while (((b = readByte()) & 0x80) != 0) {
            value |= (b & 0x7F) << i;
            i += 7;
            if (i > 35) {
                throw new IllegalArgumentException("Invalid varint");
            }
        }
        return value | (b << i);
    }

    public String readString() {
        int stringLength = readVarint();
        byte[] bytes = readBytes(stringLength);
        return new String(bytes, StandardCharsets.UTF_8);
    }

    public boolean hasNext() {
        return buffer.hasRemaining();
    }

    public void mark() {
        buffer.mark();
    }

    public void reset() {
        buffer.reset();
    }

    public ByteBuffer byteBuffer() {
        return this.buffer;
    }
}
