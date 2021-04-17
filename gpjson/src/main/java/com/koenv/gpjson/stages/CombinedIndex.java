package com.koenv.gpjson.stages;

import com.koenv.gpjson.gpu.*;
import com.koenv.gpjson.kernel.GPJSONKernel;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class CombinedIndex {
    public static final int GRID_SIZE = 8;
    public static final int BLOCK_SIZE = 1024;
    public static final int CARRY_INDEX_SIZE = GRID_SIZE * BLOCK_SIZE;
    public static final int COUNT_INDEX_SIZE = GRID_SIZE * BLOCK_SIZE;

    private final CUDARuntime cudaRuntime;
    private final ManagedGPUPointer fileMemory;

    private final long resultSize;

    public CombinedIndex(CUDARuntime cudaRuntime, ManagedGPUPointer fileMemory) {
        this.cudaRuntime = cudaRuntime;
        this.fileMemory = fileMemory;

        this.resultSize = (this.fileMemory.size() + 64 - 1) / 64;
    }

    public CombinedIndexResult create() {
        cudaRuntime.timings.start("CombinedIndex#create");
        // The string index memory is used for both the final string index and for the quote index
        ManagedGPUPointer stringIndexMemory = cudaRuntime.allocateUnmanagedMemory(resultSize, Type.SINT64);
        ManagedGPUPointer newlineIndexMemory;

        try (
                // The carry index memory is used for both the escape carry and the quote carry
                ManagedGPUPointer stringCarryIndexMemory = cudaRuntime.allocateUnmanagedMemory(CARRY_INDEX_SIZE);
                // TODO: The escape index memory can be combined with the string index memory
                ManagedGPUPointer escapeIndexMemory = cudaRuntime.allocateUnmanagedMemory(resultSize, Type.SINT64);
                ManagedGPUPointer newlineCountIndexMemory = cudaRuntime.allocateUnmanagedMemory(COUNT_INDEX_SIZE, Type.SINT32)
        ) {
            createEscapeCarryNewlineCarryIndex(stringCarryIndexMemory, newlineCountIndexMemory);

            int sum = createOffsetsIndex(newlineCountIndexMemory);
            newlineIndexMemory = cudaRuntime.allocateUnmanagedMemory(sum, Type.SINT64);

            createEscapeNewlineIndex(stringCarryIndexMemory, escapeIndexMemory, newlineCountIndexMemory, newlineIndexMemory);

            createQuoteIndex(escapeIndexMemory, stringIndexMemory, stringCarryIndexMemory);

            createStringIndex(stringIndexMemory, stringCarryIndexMemory);
        }

        cudaRuntime.timings.end();

        return new CombinedIndexResult(stringIndexMemory, newlineIndexMemory);
    }

    void createEscapeCarryNewlineCarryIndex(ManagedGPUPointer carryIndexMemory, ManagedGPUPointer newlineCountIndexMemory) {
        Kernel kernel = cudaRuntime.getKernel(GPJSONKernel.CREATE_COMBINED_ESCAPE_CARRY_NEWLINE_COUNT_INDEX);

        List<UnsafeHelper.MemoryObject> arguments = new ArrayList<>();
        arguments.add(UnsafeHelper.createPointerObject(fileMemory));
        arguments.add(UnsafeHelper.createInteger64Object(fileMemory.size()));
        arguments.add(UnsafeHelper.createPointerObject(carryIndexMemory));
        arguments.add(UnsafeHelper.createPointerObject(newlineCountIndexMemory));

        kernel.execute(new Dim3(GRID_SIZE), new Dim3(BLOCK_SIZE), 0, 0, arguments);
    }

    void createEscapeNewlineIndex(ManagedGPUPointer carryIndexMemory, ManagedGPUPointer escapeIndexMemory, ManagedGPUPointer newlineCountIndexMemory, ManagedGPUPointer newlineIndexMemory) {
        Kernel kernel = cudaRuntime.getKernel(GPJSONKernel.CREATE_COMBINED_ESCAPE_NEWLINE_INDEX);

        List<UnsafeHelper.MemoryObject> arguments = new ArrayList<>();
        arguments.add(UnsafeHelper.createPointerObject(fileMemory));
        arguments.add(UnsafeHelper.createInteger64Object(fileMemory.size()));
        arguments.add(UnsafeHelper.createPointerObject(carryIndexMemory));
        arguments.add(UnsafeHelper.createPointerObject(newlineCountIndexMemory));
        arguments.add(UnsafeHelper.createPointerObject(escapeIndexMemory));
        arguments.add(UnsafeHelper.createInteger64Object(resultSize));
        arguments.add(UnsafeHelper.createPointerObject(newlineIndexMemory));

        kernel.execute(new Dim3(GRID_SIZE), new Dim3(BLOCK_SIZE), 0, 0, arguments);
    }

    void createQuoteIndex(ManagedGPUPointer escapeIndexMemory, ManagedGPUPointer quoteIndexMemory, ManagedGPUPointer carryIndexMemory) {
        Kernel kernel = cudaRuntime.getKernel(GPJSONKernel.CREATE_QUOTE_INDEX);

        List<UnsafeHelper.MemoryObject> arguments = new ArrayList<>();
        arguments.add(UnsafeHelper.createPointerObject(fileMemory));
        arguments.add(UnsafeHelper.createInteger64Object(fileMemory.size()));
        arguments.add(UnsafeHelper.createPointerObject(escapeIndexMemory));
        arguments.add(UnsafeHelper.createPointerObject(quoteIndexMemory));
        arguments.add(UnsafeHelper.createPointerObject(carryIndexMemory));
        arguments.add(UnsafeHelper.createInteger64Object(resultSize));

        kernel.execute(new Dim3(GRID_SIZE), new Dim3(BLOCK_SIZE), 0, 0, arguments);

        cudaRuntime.timings.start("CombinedIndex#copyCarryBuffer");

        ByteBuffer carryBuffer = carryIndexMemory.copyToHost();

        byte previousValue = 0;
        while (carryBuffer.hasRemaining()) {
            carryBuffer.mark();
            byte value = (byte) (carryBuffer.get() ^ previousValue);
            carryBuffer.reset();

            carryBuffer.put(value);

            previousValue = value;
        }

        carryIndexMemory.loadFrom(carryBuffer);

        cudaRuntime.timings.end();
    }

    void createStringIndex(ManagedGPUPointer quoteIndexMemory, ManagedGPUPointer quoteCountMemory) {
        Kernel kernel = cudaRuntime.getKernel(GPJSONKernel.CREATE_STRING_INDEX);

        List<UnsafeHelper.MemoryObject> arguments = new ArrayList<>();
        arguments.add(UnsafeHelper.createInteger64Object(resultSize));
        arguments.add(UnsafeHelper.createPointerObject(quoteIndexMemory));
        arguments.add(UnsafeHelper.createPointerObject(quoteCountMemory));

        kernel.execute(new Dim3(GRID_SIZE), new Dim3(BLOCK_SIZE), 0, 0, arguments);
    }

    int createOffsetsIndex(ManagedGPUPointer countMemory) {
        cudaRuntime.timings.start("CombinedIndex#createOffsetsIndex");

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
}
