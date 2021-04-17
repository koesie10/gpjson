package com.koenv.gpjson;

import com.koenv.gpjson.debug.CUDARuntimeWrapper;
import com.koenv.gpjson.gpu.CUDAMemcpyKind;
import com.koenv.gpjson.gpu.CUDARuntime;
import com.koenv.gpjson.gpu.ManagedGPUPointer;
import com.koenv.gpjson.gpu.UnsafeHelper;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.PolyglotAccess;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;

import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

public class KernelTest {
    private static Context context = Context.newBuilder("nfi", "gpjson")
            .allowPolyglotAccess(PolyglotAccess.ALL)
            .allowNativeAccess(true)
            .build();

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

    protected byte[] readFile(String name) throws IOException {
        try (InputStream is = getClass().getClassLoader().getResourceAsStream(name)) {
            if (is == null) {
                throw new FileNotFoundException("File " + name + " not found in resources");
            }

            ByteArrayOutputStream result = new ByteArrayOutputStream();
            byte[] buffer = new byte[1 << 20];
            for (int length; (length = is.read(buffer)) != -1; ) {
                result.write(buffer, 0, length);
            }
            return result.toByteArray();
        }
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
