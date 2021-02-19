package com.koenv.gpjson.gpu;

public class ManagedGPUPointer implements AutoCloseable {
    private final CUDARuntime cudaRuntime;
    private final GPUPointer pointer;

    public ManagedGPUPointer(CUDARuntime cudaRuntime, GPUPointer pointer) {
        this.cudaRuntime = cudaRuntime;
        this.pointer = pointer;
    }

    public GPUPointer getPointer() {
        return pointer;
    }

    @Override
    public void close() {
        cudaRuntime.cudaFree(this.pointer);
    }
}
