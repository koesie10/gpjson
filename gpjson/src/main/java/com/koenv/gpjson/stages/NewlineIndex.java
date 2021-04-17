package com.koenv.gpjson.stages;

import com.koenv.gpjson.debug.GPUUtils;
import com.koenv.gpjson.gpu.*;
import com.koenv.gpjson.kernel.GPJSONKernel;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class NewlineIndex {
    public static final int GRID_SIZE = 8;
    public static final int BLOCK_SIZE = 1024;
    public static final int COUNT_INDEX_SIZE = GRID_SIZE * BLOCK_SIZE;

    private final CUDARuntime cudaRuntime;
    private final ManagedGPUPointer fileMemory;

    public NewlineIndex(CUDARuntime cudaRuntime, ManagedGPUPointer fileMemory) {
        this.cudaRuntime = cudaRuntime;
        this.fileMemory = fileMemory;
    }

    public ManagedGPUPointer create() {
        cudaRuntime.timings.start("NewlineIndex#create");
        try (ManagedGPUPointer countMemory = cudaRuntime.allocateUnmanagedMemory(COUNT_INDEX_SIZE, Type.SINT32)) {
            countNewlines(countMemory);
            int sum = createOffsetsIndex(countMemory);

            ManagedGPUPointer indexMemory = cudaRuntime.allocateUnmanagedMemory(sum, Type.SINT64);
            createIndex(countMemory, indexMemory);

            return indexMemory;
        } finally {
            cudaRuntime.timings.end();
        }
    }

    void countNewlines(ManagedGPUPointer countMemory) {
        Kernel kernel = cudaRuntime.getKernel(GPJSONKernel.COUNT_NEWLINES);

        List<UnsafeHelper.MemoryObject> arguments = new ArrayList<>();
        arguments.add(UnsafeHelper.createPointerObject(fileMemory));
        arguments.add(UnsafeHelper.createInteger64Object(fileMemory.size()));
        arguments.add(UnsafeHelper.createPointerObject(countMemory));

        kernel.execute(new Dim3(GRID_SIZE), new Dim3(BLOCK_SIZE), 0, 0, arguments);
    }

    int createOffsetsIndex(ManagedGPUPointer countMemory) {
        cudaRuntime.timings.start("NewlineIndex#createOffsetsIndex");

        ByteBuffer indexCountBuffer = countMemory.copyToHost();

        // We start at 1, the first "new-line" is at position 0
        int sum = 1;
        while (indexCountBuffer.hasRemaining()) {
            indexCountBuffer.mark();
            int value = indexCountBuffer.getInt();
            indexCountBuffer.reset();

            indexCountBuffer.putInt(sum);

            sum += value;
        }

        countMemory.loadFrom(indexCountBuffer);

        cudaRuntime.timings.end();

        return sum;
    }

    void createIndex(ManagedGPUPointer offsetMemory, ManagedGPUPointer indexMemory) {
        Kernel kernel = cudaRuntime.getKernel(GPJSONKernel.CREATE_NEWLINE_INDEX);

        List<UnsafeHelper.MemoryObject> arguments = new ArrayList<>();
        arguments.add(UnsafeHelper.createPointerObject(fileMemory));
        arguments.add(UnsafeHelper.createInteger64Object(fileMemory.size()));
        arguments.add(UnsafeHelper.createPointerObject(offsetMemory));
        arguments.add(UnsafeHelper.createPointerObject(indexMemory));

        kernel.execute(new Dim3(GRID_SIZE), new Dim3(BLOCK_SIZE), 0, 0, arguments);
    }
}
