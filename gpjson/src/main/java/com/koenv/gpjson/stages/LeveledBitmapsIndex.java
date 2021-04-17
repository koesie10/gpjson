package com.koenv.gpjson.stages;

import com.koenv.gpjson.gpu.*;
import com.koenv.gpjson.kernel.GPJSONKernel;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class LeveledBitmapsIndex {
    public static final int GRID_SIZE = 8;
    public static final int BLOCK_SIZE = 1024;

    public static final int CARRY_INDEX_SIZE = GRID_SIZE * BLOCK_SIZE;
    public static final byte MAX_NUM_LEVELS = 22;

    private final CUDARuntime cudaRuntime;
    private final ManagedGPUPointer fileMemory;
    private final ManagedGPUPointer stringIndexMemory;

    private final long levelSize;
    private final long resultSize;
    private final byte numLevels;

    public LeveledBitmapsIndex(CUDARuntime cudaRuntime, ManagedGPUPointer fileMemory, ManagedGPUPointer stringIndexMemory, byte numLevels) {
        this.cudaRuntime = cudaRuntime;
        this.fileMemory = fileMemory;
        this.stringIndexMemory = stringIndexMemory;

        if (numLevels < 0 || numLevels > MAX_NUM_LEVELS) {
            throw new IllegalArgumentException("Invalid number of levels");
        }

        this.levelSize = (this.fileMemory.size() + 64 - 1) / 64;
        this.numLevels = numLevels;
        this.resultSize = levelSize * numLevels;
    }

    public ManagedGPUPointer create() {
        // The string index memory is used for both the final string index and for the quote index
        ManagedGPUPointer leveledBitmapsIndexMemory = cudaRuntime.allocateUnmanagedMemory(resultSize, Type.SINT64);

        try (ManagedGPUPointer carryMemory = cudaRuntime.allocateUnmanagedMemory(CARRY_INDEX_SIZE, Type.SINT8)) {
            createCarryIndex(carryMemory);

            createLevelsIndex(carryMemory);

            createLeveledBitmaps(leveledBitmapsIndexMemory, carryMemory);
        }

        return leveledBitmapsIndexMemory;
    }

    void createCarryIndex(ManagedGPUPointer carryMemory) {
        Kernel kernel = cudaRuntime.getKernel(GPJSONKernel.CREATE_LEVELED_BITMAPS_CARRY_INDEX);

        List<UnsafeHelper.MemoryObject> arguments = new ArrayList<>();
        arguments.add(UnsafeHelper.createPointerObject(fileMemory));
        arguments.add(UnsafeHelper.createInteger64Object(fileMemory.size()));
        arguments.add(UnsafeHelper.createPointerObject(stringIndexMemory));
        arguments.add(UnsafeHelper.createPointerObject(carryMemory));

        kernel.execute(new Dim3(GRID_SIZE), new Dim3(BLOCK_SIZE), 0, 0, arguments);
    }

    void createLevelsIndex(ManagedGPUPointer carryMemory) {
        cudaRuntime.timings.start("LeveledBitmapsIndex#createLevelsIndex");

        ByteBuffer carryBuffer = carryMemory.copyToHost();

        byte level = -1;
        while (carryBuffer.hasRemaining()) {
            carryBuffer.mark();
            byte value = carryBuffer.get();
            carryBuffer.reset();

            carryBuffer.put(level);

            level += value;
        }

        carryMemory.loadFrom(carryBuffer);

        cudaRuntime.timings.end();
    }

    void createLeveledBitmaps(ManagedGPUPointer leveledBitmapsIndexMemory, ManagedGPUPointer carryIndexMemory) {
        Kernel kernel = cudaRuntime.getKernel(GPJSONKernel.CREATE_LEVELED_BITMAPS);

        List<UnsafeHelper.MemoryObject> arguments = new ArrayList<>();
        arguments.add(UnsafeHelper.createPointerObject(fileMemory));
        arguments.add(UnsafeHelper.createInteger64Object(fileMemory.size()));
        arguments.add(UnsafeHelper.createPointerObject(stringIndexMemory));
        arguments.add(UnsafeHelper.createPointerObject(carryIndexMemory));
        arguments.add(UnsafeHelper.createPointerObject(leveledBitmapsIndexMemory));
        arguments.add(UnsafeHelper.createInteger64Object(resultSize));
        arguments.add(UnsafeHelper.createInteger64Object(levelSize));
        arguments.add(UnsafeHelper.createInteger32Object(numLevels));

        kernel.execute(new Dim3(GRID_SIZE), new Dim3(BLOCK_SIZE), 0, 0, arguments);
    }
}
