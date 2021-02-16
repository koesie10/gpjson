package com.koenv.gpjson.gpu;

public class ManagedGPUMemory implements AutoCloseable {
    private final CUDARuntime cudaRuntime;
    private final LittleEndianNativeArrayView view;

    public ManagedGPUMemory(CUDARuntime cudaRuntime, LittleEndianNativeArrayView view) {
        this.cudaRuntime = cudaRuntime;
        this.view = view;
    }

    public LittleEndianNativeArrayView getView() {
        return view;
    }

    @Override
    public void close() {
        cudaRuntime.cudaFree(this.view);
    }
}
