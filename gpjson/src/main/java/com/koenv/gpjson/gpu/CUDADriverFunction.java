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

import com.oracle.truffle.api.interop.UnknownIdentifierException;

public enum CUDADriverFunction {
    CU_CTXCREATE("cuCtxCreate", "(pointer, uint32, sint32) :sint32"),
    CU_CTXDESTROY("cuCtxDestroy", "(pointer): sint32"),
    CU_CTXSYNCHRONIZE("cuCtxSynchronize", "(): sint32"),
    CU_DEVICEGETCOUNT("cuDeviceGetCount", "(pointer): sint32"),
    CU_DEVICEGET("cuDeviceGet", "(pointer, sint32): sint32"),
    CU_DEVICEGETNAME("cuDeviceGetName", "(pointer, sint32, sint32): sint32"),
    CU_DEVICEPRIMARYCTXRETAIN("cuDevicePrimaryCtxRetain", "(pointer, sint32): sint32"),
    CU_INIT("cuInit", "(uint32): sint32"),
    CU_LAUNCHKERNEL("cuLaunchKernel", "(uint64, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uint64, pointer, pointer): sint32"),
    CU_MODULELOAD("cuModuleLoad", "(pointer, string): sint32"),
    CU_MODULELOADDATA("cuModuleLoadData", "(pointer, string): sint32"),
    CU_MODULEUNLOAD("cuModuleUnload", "(uint64): sint32"),
    CU_MODULEGETFUNCTION("cuModuleGetFunction", "(pointer, uint64, string): sint32");

    private final String symbolName;
    private final String signature;

    CUDADriverFunction(String symbolName, String nfiSignature) {
        this.symbolName = symbolName;
        this.signature = nfiSignature;
    }

    public Object getSymbol(CUDARuntime runtime) throws UnknownIdentifierException {
        return runtime.getSymbol(CUDARuntime.CUDA_LIBRARY_NAME, symbolName, signature);
    }
}
