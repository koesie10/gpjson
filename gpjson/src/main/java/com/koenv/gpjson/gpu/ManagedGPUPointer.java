package com.koenv.gpjson.gpu;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class ManagedGPUPointer implements AutoCloseable {
    private final CUDARuntime cudaRuntime;
    private final GPUPointer pointer;
    private final long size;
    private final long numberOfElements;
    private final Type elementType;

    public ManagedGPUPointer(CUDARuntime cudaRuntime, GPUPointer pointer, long size, long numberOfElements, Type elementType) {
        this.cudaRuntime = cudaRuntime;
        this.pointer = pointer;
        this.size = size;
        this.numberOfElements = numberOfElements;
        this.elementType = elementType;
    }

    public GPUPointer getPointer() {
        return pointer;
    }

    public long size() {
        return size;
    }

    public long numberOfElements() {
        return numberOfElements;
    }

    @Override
    public void close() {
        cudaRuntime.cudaFree(this.pointer);
    }

    public void copyTo(ByteBuffer buffer) {
        UnsafeHelper.ByteArray byteArray = UnsafeHelper.createByteArray(buffer);

        cudaRuntime.cudaMemcpy(byteArray.getAddress(), pointer.getRawPointer(), buffer.capacity(), CUDAMemcpyKind.DEVICE_TO_HOST);
    }

    public ByteBuffer copyToHost() {
        ByteBuffer buffer = ByteBuffer.allocateDirect((int) size).order(ByteOrder.LITTLE_ENDIAN);

        copyTo(buffer);

        return buffer;
    }


    public ByteBuffer copyToHost(int maxSizeBytes) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(Math.min((int) size, maxSizeBytes)).order(ByteOrder.LITTLE_ENDIAN);

        copyTo(buffer);

        return buffer;
    }

    public void loadFrom(ByteBuffer buffer) {
        UnsafeHelper.ByteArray byteArray = UnsafeHelper.createByteArray(buffer);

        cudaRuntime.cudaMemcpy(pointer.getRawPointer(), byteArray.getAddress(), size, CUDAMemcpyKind.HOST_TO_DEVICE);
    }
}
