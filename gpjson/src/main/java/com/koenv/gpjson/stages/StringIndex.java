package com.koenv.gpjson.stages;

import com.koenv.gpjson.gpu.*;
import com.koenv.gpjson.kernel.GPJSONKernel;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class StringIndex {
    public static final int GRID_SIZE = 8;
    public static final int BLOCK_SIZE = 1024;
    public static final int CARRY_INDEX_SIZE = GRID_SIZE * BLOCK_SIZE;

    private final CUDARuntime cudaRuntime;
    private final ManagedGPUPointer fileMemory;

    private final long resultSize;

    public StringIndex(CUDARuntime cudaRuntime, ManagedGPUPointer fileMemory) {
        this.cudaRuntime = cudaRuntime;
        this.fileMemory = fileMemory;

        this.resultSize = (this.fileMemory.size() + 64 - 1) / 64;
    }

    public ManagedGPUPointer create() {
        // The string index memory is used for both the final string index and for the quote index
        ManagedGPUPointer stringIndexMemory = cudaRuntime.allocateUnmanagedMemory(resultSize, Type.SINT64);

        try (
                // The carry index memory is used for both the escape carry and the quote carry
                ManagedGPUPointer carryIndexMemory = cudaRuntime.allocateUnmanagedMemory(CARRY_INDEX_SIZE);
                // TODO: The escape index memory can be combined with the string index memory
                ManagedGPUPointer escapeIndexMemory = cudaRuntime.allocateUnmanagedMemory(resultSize, Type.SINT64)
        ) {
            createEscapeCarryIndex(carryIndexMemory);
            createEscapeIndex(carryIndexMemory, escapeIndexMemory);

            createQuoteIndex(escapeIndexMemory, stringIndexMemory, carryIndexMemory);

            createStringIndex(stringIndexMemory, carryIndexMemory);
        }

        return stringIndexMemory;
    }

    void createEscapeCarryIndex(ManagedGPUPointer carryIndexMemory) {
        Kernel kernel = cudaRuntime.getKernel(GPJSONKernel.CREATE_ESCAPE_CARRY_INDEX);

        List<UnsafeHelper.MemoryObject> arguments = new ArrayList<>();
        arguments.add(UnsafeHelper.createPointerObject(fileMemory.getPointer().getRawPointer()));
        arguments.add(UnsafeHelper.createInteger64Object(fileMemory.size()));
        arguments.add(UnsafeHelper.createPointerObject(carryIndexMemory.getPointer().getRawPointer()));

        kernel.execute(new Dim3(GRID_SIZE), new Dim3(BLOCK_SIZE), 0, 0, arguments);
    }

    void createEscapeIndex(ManagedGPUPointer carryIndexMemory, ManagedGPUPointer escapeIndexMemory) {
        Kernel kernel = cudaRuntime.getKernel(GPJSONKernel.CREATE_ESCAPE_INDEX);

        List<UnsafeHelper.MemoryObject> arguments = new ArrayList<>();
        arguments.add(UnsafeHelper.createPointerObject(fileMemory));
        arguments.add(UnsafeHelper.createInteger64Object(fileMemory.size()));
        arguments.add(UnsafeHelper.createPointerObject(carryIndexMemory));
        arguments.add(UnsafeHelper.createPointerObject(escapeIndexMemory));
        arguments.add(UnsafeHelper.createInteger64Object(resultSize));

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

        ByteBuffer carryBuffer = carryIndexMemory.copyToHost();

        byte previousValue = 0;
        while (carryBuffer.hasRemaining()) {
            carryBuffer.mark();
            byte value = (byte) (carryBuffer.get() ^ previousValue);
            carryBuffer.reset();

            carryBuffer.put(value);
        }

        carryIndexMemory.loadFrom(carryBuffer);
    }

    void createStringIndex(ManagedGPUPointer quoteIndexMemory, ManagedGPUPointer quoteCountMemory) {
        Kernel kernel = cudaRuntime.getKernel(GPJSONKernel.CREATE_STRING_INDEX);

        List<UnsafeHelper.MemoryObject> arguments = new ArrayList<>();
        arguments.add(UnsafeHelper.createInteger64Object(resultSize));
        arguments.add(UnsafeHelper.createPointerObject(quoteIndexMemory));
        arguments.add(UnsafeHelper.createPointerObject(quoteCountMemory));

        kernel.execute(new Dim3(GRID_SIZE), new Dim3(BLOCK_SIZE), 0, 0, arguments);
    }
}
