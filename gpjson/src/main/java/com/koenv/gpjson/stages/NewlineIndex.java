package com.koenv.gpjson.stages;

import com.koenv.gpjson.gpu.*;
import com.koenv.gpjson.kernel.GPJSONKernel;

import java.nio.ByteBuffer;
import java.util.ArrayList;
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

    public long[] create() {
        try (ManagedGPUPointer countMemory = cudaRuntime.allocateUnmanagedMemory(COUNT_INDEX_SIZE, Type.SINT32)) {
            countNewlines(countMemory);
            int sum = createOffsetsIndex(countMemory);

            try (ManagedGPUPointer indexMemory = cudaRuntime.allocateUnmanagedMemory(sum, Type.SINT64)) {
                return createIndex(sum, countMemory, indexMemory);
            }
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
        ByteBuffer indexCountBuffer = countMemory.copyToHost();

        int sum = 0;
        while (indexCountBuffer.hasRemaining()) {
            indexCountBuffer.mark();
            int value = indexCountBuffer.getInt();
            indexCountBuffer.reset();

            indexCountBuffer.putInt(sum);

            sum += value;
        }

        countMemory.loadFrom(indexCountBuffer);

        return sum;
    }

    long[] createIndex(int sum, ManagedGPUPointer offsetMemory, ManagedGPUPointer indexMemory) {
        Kernel kernel = cudaRuntime.getKernel(GPJSONKernel.CREATE_NEWLINE_INDEX);

        List<UnsafeHelper.MemoryObject> arguments = new ArrayList<>();
        arguments.add(UnsafeHelper.createPointerObject(fileMemory));
        arguments.add(UnsafeHelper.createInteger64Object(fileMemory.size()));
        arguments.add(UnsafeHelper.createPointerObject(offsetMemory));
        arguments.add(UnsafeHelper.createPointerObject(indexMemory));

        kernel.execute(new Dim3(GRID_SIZE), new Dim3(BLOCK_SIZE), 0, 0, arguments);

        ByteBuffer indexBuffer = indexMemory.copyToHost();

        long[] index = new long[sum];
        int i = 0;
        while (indexBuffer.hasRemaining()) {
            index[i++] = indexBuffer.getLong();
        }

        return index;
    }
}
