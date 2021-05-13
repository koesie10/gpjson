package com.koenv.gpjson.jsonpath;

import java.nio.ByteBuffer;

public interface IRVisitor {
    void visitProperty(String name);

    void visitIndex(int index);

    void visitStoreResult();

    void visitDown();

    void visitUp();

    void visitEnd();

    static void accept(byte[] ir, IRVisitor visitor) {
        accept(new IRByteInputBuffer(ByteBuffer.wrap(ir)), visitor);
    }

    static void accept(IRByteInputBuffer buffer, IRVisitor visitor) {
        while (buffer.hasNext()) {
            switch (buffer.readOpcode()) {
                case END:
                    visitor.visitEnd();
                    break;
                case STORE_RESULT:
                    visitor.visitStoreResult();
                    break;
                case MOVE_DOWN:
                    visitor.visitDown();
                    break;
                case MOVE_UP:
                    visitor.visitUp();
                    break;
                case MOVE_TO_KEY:
                    visitor.visitProperty(buffer.readString());
                    break;
                case MOVE_TO_INDEX:
                    visitor.visitIndex(buffer.readVarint());
                    break;
            }
        }
    }
}
