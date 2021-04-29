package com.koenv.gpjson;

import com.koenv.gpjson.debug.CUDARuntimeWrapper;
import com.koenv.gpjson.gpu.CUDAMemcpyKind;
import com.koenv.gpjson.gpu.CUDARuntime;
import com.koenv.gpjson.gpu.ManagedGPUPointer;
import com.koenv.gpjson.gpu.UnsafeHelper;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;

import java.io.IOException;
import java.nio.ByteBuffer;

public class KernelTest extends GPJSONTest {
    protected CUDARuntime cudaRuntime;

    @BeforeEach
    void setUp() {
        context.enter();

        CUDARuntimeWrapper wrapper = new CUDARuntimeWrapper(null);
        context.eval("gpjson", "jsonpath").invokeMember("UNSAFE_getCUDARuntime", wrapper);

        cudaRuntime = wrapper.getCudaRuntime();
    }

    @AfterEach
    public void tearDown() {
        context.leave();
    }

    protected ManagedGPUPointer readFileToGPU(String name) throws IOException {
        byte[] bytes = readFile(name);
        long size = bytes.length;

        ByteBuffer fileBuffer = ByteBuffer.allocateDirect((int) size);
        fileBuffer.put(bytes);
        UnsafeHelper.ByteArray fileByteArray = UnsafeHelper.createByteArray(fileBuffer);

        ManagedGPUPointer fileMemory = cudaRuntime.allocateUnmanagedMemory(fileBuffer.capacity());

        cudaRuntime.cudaMemcpy(fileMemory.getPointer().getRawPointer(), fileByteArray.getAddress(), fileBuffer.capacity(), CUDAMemcpyKind.HOST_TO_DEVICE);

        return fileMemory;
    }
}
