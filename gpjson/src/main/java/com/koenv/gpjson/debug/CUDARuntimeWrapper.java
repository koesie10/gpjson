package com.koenv.gpjson.debug;

import com.koenv.gpjson.gpu.CUDARuntime;
import com.oracle.truffle.api.interop.TruffleObject;

public class CUDARuntimeWrapper implements TruffleObject {
    private CUDARuntime cudaRuntime;

    public CUDARuntimeWrapper(CUDARuntime cudaRuntime) {
        this.cudaRuntime = cudaRuntime;
    }

    public CUDARuntime getCudaRuntime() {
        return cudaRuntime;
    }

    public void setCudaRuntime(CUDARuntime cudaRuntime) {
        this.cudaRuntime = cudaRuntime;
    }
}
