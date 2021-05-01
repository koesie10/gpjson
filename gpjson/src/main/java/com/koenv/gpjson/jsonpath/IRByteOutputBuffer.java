package com.koenv.gpjson.jsonpath;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

public class IRByteOutputBuffer {
    private final ByteArrayOutputStream buffer;

    public IRByteOutputBuffer() {
        this.buffer = new ByteArrayOutputStream();
    }

    public IRByteOutputBuffer writeByte(byte b) {
        buffer.write(b);
        return this;
    }

    public IRByteOutputBuffer writeByte(int b) {
        this.writeByte((byte) (b & 0xFF));
        return this;
    }

    public IRByteOutputBuffer writeBytes(byte[] b) {
        try {
            buffer.write(b);
        } catch (IOException e) {
            // This should never happen for the ByteArrayOutputStream
            throw new RuntimeException(e);
        }
        return this;
    }

    public IRByteOutputBuffer writeVarInt(int i) {
        while((i & 0xFFFFFF80) != 0) {
            this.writeByte(i & 0x7F | 0x80);
            i >>>= 7;
        }

        this.writeByte(i & 0x7F);
        return this;
    }

    public IRByteOutputBuffer writeString(String s) {
        byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
        this.writeVarInt(bytes.length);
        this.writeBytes(bytes);
        return this;
    }

    public int size() {
        return buffer.size();
    }

    public byte[] toByteArray() {
        return buffer.toByteArray();
    }

    public ByteBuffer toByteBuffer() {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(size());
        byteBuffer.put(toByteArray());
        return byteBuffer;
    }

    public ByteBuffer toWrappedByteBuffer() {
        return ByteBuffer.wrap(this.toByteArray());
    }

    public IRByteInputBuffer toReadable() {
        return new IRByteInputBuffer(toWrappedByteBuffer());
    }
}
