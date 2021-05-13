package com.koenv.gpjson.jsonpath;

public class IRBuilder {
    private final IRByteOutputBuffer buffer;

    private int currentLevel;
    private int numResultStores;

    private boolean ended;

    public IRBuilder() {
        this(new IRByteOutputBuffer());
    }

    public IRBuilder(IRByteOutputBuffer buffer) {
        this.buffer = buffer;
    }

    public IRBuilder property(String name) {
        buffer.writeOpcode(Opcode.MOVE_TO_KEY);
        buffer.writeString(name);

        return this;
    }

    public IRBuilder index(int index) {
        buffer.writeOpcode(Opcode.MOVE_TO_INDEX);
        buffer.writeVarInt(index);

        return this;
    }

    public IRBuilder down() {
        buffer.writeOpcode(Opcode.MOVE_DOWN);

        currentLevel++;

        return this;
    }

    public IRBuilder up() {
        buffer.writeOpcode(Opcode.MOVE_UP);

        currentLevel--;

        return this;
    }

    public IRBuilder storeResult() {
        buffer.writeOpcode(Opcode.STORE_RESULT);

        numResultStores++;

        return this;
    }

    public IRBuilder end() {
        if (ended) {
            throw new IllegalStateException("IR has already been ended");
        }

        ended = true;
        buffer.writeOpcode(Opcode.END);

        return this;
    }

    public int getCurrentLevel() {
        return currentLevel;
    }

    public int getNumResultStores() {
        return numResultStores;
    }

    public IRByteOutputBuffer getBuffer() {
        return buffer;
    }

    public byte[] toByteArray() {
        if (!ended) {
            throw new IllegalStateException("Cannot convert to byte array until end() has been called");
        }

        return buffer.toByteArray();
    }
}
