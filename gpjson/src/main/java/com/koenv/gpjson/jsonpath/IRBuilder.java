package com.koenv.gpjson.jsonpath;

public class IRBuilder {
    private final IRByteBuffer buffer;

    private boolean ended;

    public IRBuilder() {
        this(new IRByteBuffer());
    }

    public IRBuilder(IRByteBuffer buffer) {
        this.buffer = buffer;
    }

    public IRBuilder property(String name) {
        buffer.writeByte(0x01);
        buffer.writeString(name);

        return this;
    }

    public IRBuilder index(int index) {
        buffer.writeByte(0x02);
        buffer.writeVarInt(index);

        return this;
    }

    public IRBuilder end() {
        if (ended) {
            throw new IllegalStateException("IR has already been ended");
        }

        ended = true;
        buffer.writeByte(0x00);

        return this;
    }

    public IRByteBuffer getBuffer() {
        return buffer;
    }

    public byte[] toByteArray() {
        if (!ended) {
            throw new IllegalStateException("Cannot convert to byte array until end() has been called");
        }

        return buffer.toByteArray();
    }
}
