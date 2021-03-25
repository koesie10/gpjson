/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, 2020, Oracle and/or its affiliates. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
package com.koenv.gpjson.gpu;

import com.oracle.truffle.api.CompilerDirectives;
import sun.misc.Unsafe;

import java.io.ByteArrayOutputStream;
import java.lang.reflect.Field;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;

public class UnsafeHelper {
    private static final Unsafe unsafe;
    private static final long bufferAddressOffset;

    static {
        try {
            Field f = Unsafe.class.getDeclaredField("theUnsafe");
            f.setAccessible(true);
            unsafe = (Unsafe) f.get(null);
            bufferAddressOffset = unsafe.objectFieldOffset(Buffer.class.getDeclaredField("address"));
        } catch (NoSuchFieldException | IllegalAccessException e) {
            // this needs to be a RuntimeException since it is raised during static initialization
            throw new RuntimeException(e);
        }
    }

    public static Unsafe getUnsafe() {
        return unsafe;
    }

    public static PointerObject createPointerObject() {
        return new PointerObject();
    }

    public static PointerObject createPointerObject(long value) {
        PointerObject pointer = createPointerObject();
        pointer.setValueOfPointer(value);
        return pointer;
    }

    public static PointerObject createPointerObject(GPUPointer pointer) {
        return createPointerObject(pointer.getRawPointer());
    }

    public static PointerObject createPointerObject(ManagedGPUPointer pointer) {
        return createPointerObject(pointer.getPointer());
    }

    public static Integer8Object createInteger8Object() {
        return new Integer8Object();
    }

    public static Integer8Object createInteger8Object(byte value) {
        Integer8Object int8 = createInteger8Object();
        int8.setValue(value);
        return int8;
    }

    public static Integer16Object createInteger16Object() {
        return new Integer16Object();
    }

    public static Integer32Object createInteger32Object() {
        return new Integer32Object();
    }

    public static Integer32Object createInteger32Object(int value) {
        Integer32Object int32 = createInteger32Object();
        int32.setValue(value);
        return int32;
    }

    public static Integer64Object createInteger64Object() {
        return new Integer64Object();
    }

    public static Integer64Object createInteger64Object(long value) {
        Integer64Object int32 = createInteger64Object();
        int32.setValue(value);
        return int32;
    }

    public static Float32Object createFloat32Object() {
        return new Float32Object();
    }

    public static Float64Object createFloat64Object() {
        return new Float64Object();
    }

    public static ByteArray createByteArray(int numElements) {
        return new ByteArray(numElements);
    }

    public static ByteArray createByteArray(ByteBuffer byteBuffer) {
        long address = unsafe.getLong(byteBuffer, bufferAddressOffset);

        return new ByteArray(address, byteBuffer.capacity());
    }

    public static PointerArray createPointerArray(int numElements) {
        return new PointerArray(numElements);
    }

    public static StringObject createStringObject(int numBytes) {
        return new StringObject(numBytes);
    }

    abstract public static class MemoryObject implements java.io.Closeable {
        private final long address;

        MemoryObject(long address) {
            this.address = address;
        }

        public final long getAddress() {
            return address;
        }

        @Override
        public void close() {
            unsafe.freeMemory(address);
        }
    }

    static final class PointerObject extends MemoryObject {

        PointerObject() {
            super(unsafe.allocateMemory(unsafe.addressSize()));
        }

        long getValueOfPointer() {
            return unsafe.getLong(getAddress());
        }

        void setValueOfPointer(long pointerValue) {
            unsafe.putLong(getAddress(), pointerValue);
        }
    }

    public static final class ByteArray extends MemoryObject {

        private final int numElements;

        ByteArray(int numElements) {
            super(unsafe.allocateMemory(numElements));
            this.numElements = numElements;
        }

        ByteArray(long address, int numElements) {
            super(address);
            this.numElements = numElements;
        }

        public void setValueAt(int index, byte value) {
            if ((index < 0) || (index >= numElements)) {
                CompilerDirectives.transferToInterpreter();
                throw new IllegalArgumentException(index + " is out of range");
            }
            unsafe.putByte(getAddress() + index, value);
        }
    }

    public static final class PointerArray extends MemoryObject {

        private final int numElements;

        PointerArray(int numElements) {
            super(unsafe.allocateMemory(unsafe.addressSize() * numElements));
            this.numElements = numElements;
        }

        public void setValueAt(int index, long pointerValue) {
            if ((index < 0) || (index >= numElements)) {
                CompilerDirectives.transferToInterpreter();
                throw new IllegalArgumentException(index + " is out of range");
            }
            unsafe.putAddress(getAddress() + index * unsafe.addressSize(), pointerValue);
        }
    }

    static final class StringObject extends MemoryObject {
        private final int maxLength;

        StringObject(int numBytes) {
            super(unsafe.allocateMemory(numBytes));
            maxLength = numBytes;
        }

        @CompilerDirectives.TruffleBoundary
        static StringObject fromJavaString(String javaString) {
            byte[] bytes = javaString.getBytes(Charset.forName("ISO-8859-1"));
            StringObject so = new StringObject(bytes.length + 1); // + 1 for \NULL terminator
            for (int i = 0; i < bytes.length; i++) {
                unsafe.putByte(so.getAddress() + i, bytes[i]);
            }
            unsafe.putByte(so.getAddress() + bytes.length, (byte) 0);  // set \null terminator
            return so;
        }

        @CompilerDirectives.TruffleBoundary
        public static String getUncheckedZeroTerminatedString(long address) {
            ByteArrayOutputStream byteStream = new ByteArrayOutputStream();
            int offset = 0;
            while (true) {
                byte b = unsafe.getByte(address + offset);
                if (b == 0) {
                    break;
                }
                byteStream.write(b);
                offset += 1;
            }
            return new String(byteStream.toByteArray(), 0, offset, Charset.forName("ISO-8859-1"));
        }

        @CompilerDirectives.TruffleBoundary
        String getZeroTerminatedString() {
            byte[] bytes = new byte[maxLength];
            int offset = 0;
            while (offset < maxLength) {
                byte b = unsafe.getByte(getAddress() + offset);
                if (b == 0) {
                    break;
                }
                bytes[offset] = b;
                offset += 1;
            }
            return new String(bytes, 0, offset, Charset.forName("ISO-8859-1"));
        }
    }

    public static final class Integer8Object extends MemoryObject {

        Integer8Object() {
            super(unsafe.allocateMemory(1));
        }

        public byte getValue() {
            return unsafe.getByte(getAddress());
        }

        public char getChar() {
            return unsafe.getChar(getAddress());
        }

        public void setValue(byte value) {
            unsafe.putByte(getAddress(), value);
        }
    }

    public static final class Integer16Object extends MemoryObject {

        Integer16Object() {
            super(unsafe.allocateMemory(2));
        }

        public short getValue() {
            return unsafe.getShort(getAddress());
        }

        public void setValue(short value) {
            unsafe.putShort(getAddress(), value);
        }
    }

    public static final class Integer32Object extends MemoryObject {

        Integer32Object() {
            super(unsafe.allocateMemory(4));
        }

        public int getValue() {
            return unsafe.getInt(getAddress());
        }

        public void setValue(int value) {
            unsafe.putInt(getAddress(), value);
        }
    }

    public static final class Integer64Object extends MemoryObject {

        Integer64Object() {
            super(unsafe.allocateMemory(8));
        }

        public long getValue() {
            return unsafe.getLong(getAddress());
        }

        public void setValue(long value) {
            unsafe.putLong(getAddress(), value);
        }
    }

    public static final class Float64Object extends MemoryObject {

        Float64Object() {
            super(unsafe.allocateMemory(8));
        }

        public double getValue() {
            return unsafe.getDouble(getAddress());
        }

        public void setValue(double value) {
            unsafe.putDouble(getAddress(), value);
        }
    }

    public static final class Float32Object extends MemoryObject {

        Float32Object() {
            super(unsafe.allocateMemory(4));
        }

        public float getValue() {
            return unsafe.getFloat(getAddress());
        }

        public void setValue(float value) {
            unsafe.putFloat(getAddress(), value);
        }
    }
}
