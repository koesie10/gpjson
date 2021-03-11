package com.koenv.gpjson.gpu;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class ManagedGPUPointer implements AutoCloseable {
    private final CUDARuntime cudaRuntime;
    private final GPUPointer pointer;
    private final long size;

    public ManagedGPUPointer(CUDARuntime cudaRuntime, GPUPointer pointer, long size) {
        this.cudaRuntime = cudaRuntime;
        this.pointer = pointer;
        this.size = size;
    }

    public GPUPointer getPointer() {
        return pointer;
    }

    public long size() {
        return size;
    }

    @Override
    public void close() {
        cudaRuntime.cudaFree(this.pointer);
    }

    public void copyTo(ByteBuffer buffer) {
        UnsafeHelper.ByteArray byteArray = UnsafeHelper.createByteArray(buffer);

        cudaRuntime.cudaMemcpy(byteArray.getAddress(), pointer.getRawPointer(), size, CUDAMemcpyKind.DEVICE_TO_HOST);
    }

    public ByteBuffer copyToHost() {
        ByteBuffer buffer = ByteBuffer.allocateDirect((int) size).order(ByteOrder.LITTLE_ENDIAN);

        copyTo(buffer);

        return buffer;
    }

    public void loadFrom(ByteBuffer buffer) {
        UnsafeHelper.ByteArray byteArray = UnsafeHelper.createByteArray(buffer);

        cudaRuntime.cudaMemcpy(pointer.getRawPointer(), byteArray.getAddress(), size, CUDAMemcpyKind.HOST_TO_DEVICE);
    }
}
