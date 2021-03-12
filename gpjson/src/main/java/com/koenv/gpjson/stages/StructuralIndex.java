package com.koenv.gpjson.stages;

import com.koenv.gpjson.gpu.*;
import com.koenv.gpjson.kernel.GPJSONKernel;

import java.util.ArrayList;
import java.util.List;

public class StructuralIndex {
    public static final int GRID_SIZE = 8;
    public static final int BLOCK_SIZE = 1024;

    private final CUDARuntime cudaRuntime;
    private final ManagedGPUPointer fileMemory;
    private final ManagedGPUPointer stringIndexMemory;

    public StructuralIndex(CUDARuntime cudaRuntime, ManagedGPUPointer fileMemory, ManagedGPUPointer stringIndexMemory) {
        this.cudaRuntime = cudaRuntime;
        this.fileMemory = fileMemory;
        this.stringIndexMemory = stringIndexMemory;
    }

    public ManagedGPUPointer create() {
        // The string index memory is used for both the final string index and for the quote index
        ManagedGPUPointer structuralIndexMemory = cudaRuntime.allocateUnmanagedMemory(fileMemory.size(), Type.CHAR);

        createStructuralIndex(structuralIndexMemory);

        return structuralIndexMemory;
    }

    void createStructuralIndex(ManagedGPUPointer structuralIndexMemory) {
        Kernel kernel = cudaRuntime.getKernel(GPJSONKernel.DISCOVER_STRUCTURE);

        List<UnsafeHelper.MemoryObject> arguments = new ArrayList<>();
        arguments.add(UnsafeHelper.createPointerObject(fileMemory));
        arguments.add(UnsafeHelper.createInteger64Object(fileMemory.size()));
        arguments.add(UnsafeHelper.createPointerObject(stringIndexMemory));
        arguments.add(UnsafeHelper.createPointerObject(structuralIndexMemory));

        kernel.execute(new Dim3(GRID_SIZE), new Dim3(BLOCK_SIZE), 0, 0, arguments);
    }
}
