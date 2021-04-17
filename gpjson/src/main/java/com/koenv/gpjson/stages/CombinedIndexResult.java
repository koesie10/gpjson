package com.koenv.gpjson.stages;

import com.koenv.gpjson.gpu.ManagedGPUPointer;

public class CombinedIndexResult implements AutoCloseable {
    public final ManagedGPUPointer stringIndex;
    public final ManagedGPUPointer newlineIndex;

    public CombinedIndexResult(ManagedGPUPointer stringIndex, ManagedGPUPointer newlineIndex) {
        this.stringIndex = stringIndex;
        this.newlineIndex = newlineIndex;
    }

    @Override
    public void close() {
        stringIndex.close();
        newlineIndex.close();
    }
}
