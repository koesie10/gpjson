package com.koenv.gpjson.jsonpath;

import java.nio.ByteBuffer;

public interface IRVisitor {
    void visitProperty(String name);

    void visitIndex(int index);

    void visitEnd();

    static void accept(byte[] ir, IRVisitor visitor) {
        accept(new IRByteInputBuffer(ByteBuffer.wrap(ir)), visitor);
    }

    static void accept(IRByteInputBuffer buffer, IRVisitor visitor) {
        while (buffer.hasNext()) {
            switch (buffer.readByte()) {
                case 0x00:
                    visitor.visitEnd();
                    break;
                case 0x01:
                    visitor.visitProperty(buffer.readString());
                    break;
                case 0x02:
                    visitor.visitIndex(buffer.readVarint());
                    break;
            }
        }
    }
}
