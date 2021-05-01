/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, 2020, Oracle and/or its affiliates. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
package com.koenv.gpjson.gpu;

import com.koenv.gpjson.GPJSONException;
import com.koenv.gpjson.util.FormatUtil;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.TruffleObject;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

public final class Kernel implements TruffleObject {

    private final CUDARuntime cudaRuntime;
    private final String kernelName;
    private final String kernelSymbol;
    private final long nativeKernelFunctionHandle;
    private final CUModule module;
    private final Parameter[] kernelParameters;
    private int launchCount = 0;
    private final String ptxCode;

    /**
     * Create a kernel and hold on to the PTX code.
     *
     * @param cudaRuntime captured reference to the CUDA runtime instance
     * @param kernelName name of kernel as exposed through Truffle
     * @param kernelSymbol name of the kernel symbol
     * @param kernelFunction native pointer to the kernel function (CUfunction)
     * @param kernelSignature signature string of the kernel (NFI or NIDL)
     * @param module CUmodule that contains the kernel function
     * @param ptx PTX source code for the kernel.
     */
    public Kernel(CUDARuntime cudaRuntime, String kernelName, String kernelSymbol,
                  long kernelFunction, String kernelSignature, CUModule module, String ptx) {
        try {
            List<Parameter> paramList = Parameter.parseParameterSignature(kernelSignature);
            Parameter[] params = new Parameter[paramList.size()];
            this.kernelParameters = paramList.toArray(params);
        } catch (GPJSONException e) {
            CompilerDirectives.transferToInterpreter();
            throw e;
        }
        this.cudaRuntime = cudaRuntime;
        this.kernelName = kernelName;
        this.kernelSymbol = kernelSymbol;
        this.nativeKernelFunctionHandle = kernelFunction;
        this.module = module;
        this.ptxCode = ptx;
    }

    public void incrementLaunchCount() {
        launchCount++;
    }

    public CUDARuntime getCudaRuntime() {
        return cudaRuntime;
    }

    public Parameter[] getKernelParameters() {
        return kernelParameters;
    }

    public long getKernelFunctionHandle() {
        if (module.isClosed()) {
            CompilerDirectives.transferToInterpreter();
            throw new GPJSONException("CUmodule containing kernel " + kernelName + " is already closed");
        }
        return nativeKernelFunctionHandle;
    }

    public CUModule getModule() {
        return module;
    }

    @Override
    public String toString() {
        return "Kernel(" + kernelName + ", " + Arrays.toString(kernelParameters) + ", launchCount=" + launchCount + ")";
    }

    public String getPTX() {
        return ptxCode;
    }

    public String getKernelName() {
        return kernelName;
    }

    public String getSymbolName() {
        return kernelSymbol;
    }

    public int getLaunchCount() {
        return launchCount;
    }

    public void execute(Dim3 gridSize, Dim3 blockSize, int dynamicSharedMemoryBytes, int stream, UnsafeHelper.MemoryObject... arguments) {
        execute(gridSize, blockSize, dynamicSharedMemoryBytes, stream, Arrays.asList(arguments));
    }

    public void execute(Dim3 gridSize, Dim3 blockSize, int dynamicSharedMemoryBytes, int stream, List<UnsafeHelper.MemoryObject> arguments) {
        cudaRuntime.timings.start("kernel#execute", kernelName);
        incrementLaunchCount();
        try (KernelArguments args = new KernelArguments(arguments)) {
            cudaRuntime.cuLaunchKernel(this, gridSize, blockSize, dynamicSharedMemoryBytes, stream, args);
        }
        cudaRuntime.timings.end();
    }

    public void executeAsync(Dim3 gridSize, Dim3 blockSize, int dynamicSharedMemoryBytes, int stream, List<UnsafeHelper.MemoryObject> arguments) {
        cudaRuntime.timings.start("kernel#executeAsync", kernelName);
        incrementLaunchCount();
        try (KernelArguments args = new KernelArguments(arguments)) {
            cudaRuntime.cuLaunchKernelAsync(this, gridSize, blockSize, dynamicSharedMemoryBytes, stream, args);
        }
        cudaRuntime.timings.end();
    }
}
