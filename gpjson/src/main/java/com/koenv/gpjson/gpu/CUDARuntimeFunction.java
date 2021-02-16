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
import com.oracle.truffle.api.CompilerAsserts;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.*;

import static com.koenv.gpjson.functions.Function.*;

public enum CUDARuntimeFunction implements CUDAFunction.Spec {
    CUDA_DEVICEGETATTRIBUTE("cudaDeviceGetAttribute", "(pointer, sint32, sint32): sint32") {
        @Override
        public Object call(CUDARuntime cudaRuntime, Object[] args) throws InteropException {
            checkArgumentLength(args, 2);
            int attributeCode = expectInt(args[0]);
            int deviceId = expectInt(args[1]);
            try (UnsafeHelper.Integer32Object value = UnsafeHelper.createInteger32Object()) {
                callSymbol(cudaRuntime, value.getAddress(), attributeCode, deviceId);
                return value.getValue();
            }
        }
    },
    CUDA_DEVICERESET("cudaDeviceReset", "(): sint32") {
        @Override
        public Object call(CUDARuntime cudaRuntime, Object[] args) throws InteropException {
            checkArgumentLength(args, 0);
            callSymbol(cudaRuntime);
            return NoneValue.get();
        }
    },
    CUDA_DEVICESYNCHRONIZE("cudaDeviceSynchronize", "(): sint32") {
        @Override
        public Object call(CUDARuntime cudaRuntime, Object[] args) throws InteropException {
            checkArgumentLength(args, 0);
            callSymbol(cudaRuntime);
            return NoneValue.get();
        }
    },
    CUDA_FREE("cudaFree", "(pointer): sint32") {
        @Override
        public Object call(CUDARuntime cudaRuntime, Object[] args) throws InteropException {
            checkArgumentLength(args, 1);
            Object pointerObj = args[0];
            long addr;
            if (pointerObj instanceof GPUPointer) {
                addr = ((GPUPointer) pointerObj).getRawPointer();
            } else if (pointerObj instanceof LittleEndianNativeArrayView) {
                addr = ((LittleEndianNativeArrayView) pointerObj).getStartAddress();
            } else {
                throw new GPJSONException("expected GPUPointer or LittleEndianNativeArrayView");
            }
            callSymbol(cudaRuntime, addr);
            return NoneValue.get();
        }
    },
    CUDA_GETDEVICE("cudaGetDevice", "(pointer): sint32") {
        @Override
        @CompilerDirectives.TruffleBoundary
        public Object call(CUDARuntime cudaRuntime, Object[] args) throws InteropException {
            checkArgumentLength(args, 0);
            try (UnsafeHelper.Integer32Object deviceId = UnsafeHelper.createInteger32Object()) {
                callSymbol(cudaRuntime, deviceId.getAddress());
                return deviceId.getValue();
            }
        }
    },
    CUDA_GETDEVICECOUNT("cudaGetDeviceCount", "(pointer): sint32") {
        @Override
        @CompilerDirectives.TruffleBoundary
        public Object call(CUDARuntime cudaRuntime, Object[] args) throws InteropException {
            checkArgumentLength(args, 0);
            try (UnsafeHelper.Integer32Object deviceCount = UnsafeHelper.createInteger32Object()) {
                callSymbol(cudaRuntime, deviceCount.getAddress());
                return deviceCount.getValue();
            }
        }
    },
    CUDA_GETERRORSTRING("cudaGetErrorString", "(sint32): string") {
        @Override
        @CompilerDirectives.TruffleBoundary
        public String call(CUDARuntime cudaRuntime, Object[] args) throws InteropException {
            checkArgumentLength(args, 1);
            int errorCode = expectInt(args[0]);
            Object result = CUDARuntime.INTEROP.execute(getSymbol(cudaRuntime), errorCode);
            return CUDARuntime.INTEROP.asString(result);
        }
    },
    CUDA_MALLOC("cudaMalloc", "(pointer, uint64): sint32") {
        @Override
        @CompilerDirectives.TruffleBoundary
        public Object call(CUDARuntime cudaRuntime, Object[] args) throws InteropException {
            checkArgumentLength(args, 1);
            long numBytes = expectLong(args[0]);
            try (UnsafeHelper.PointerObject outPointer = UnsafeHelper.createPointerObject()) {
                callSymbol(cudaRuntime, outPointer.getAddress(), numBytes);
                long addressAllocatedMemory = outPointer.getValueOfPointer();
                return new GPUPointer(addressAllocatedMemory);
            }
        }
    },
    CUDA_MALLOCMANAGED("cudaMallocManaged", "(pointer, uint64, uint32): sint32") {
        @Override
        @CompilerDirectives.TruffleBoundary
        public Object call(CUDARuntime cudaRuntime, Object[] args) throws InteropException {
            checkArgumentLength(args, 1);
            final int cudaMemAttachGlobal = 0x01;
            long numBytes = expectLong(args[0]);
            try (UnsafeHelper.PointerObject outPointer = UnsafeHelper.createPointerObject()) {
                callSymbol(cudaRuntime, outPointer.getAddress(), numBytes, cudaMemAttachGlobal);
                long addressAllocatedMemory = outPointer.getValueOfPointer();
                return new GPUPointer(addressAllocatedMemory);
            }
        }
    },
    CUDA_SETDEVICE("cudaSetDevice", "(sint32): sint32") {
        @Override
        @CompilerDirectives.TruffleBoundary
        public Object call(CUDARuntime cudaRuntime, Object[] args) throws InteropException {
            checkArgumentLength(args, 1);
            int device = expectInt(args[0]);
            callSymbol(cudaRuntime, device);
            return NoneValue.get();
        }
    },
    CUDA_MEMCPY("cudaMemcpy", "(pointer, pointer, uint64, sint32): sint32") {
        @Override
        @CompilerDirectives.TruffleBoundary
        public Object call(CUDARuntime cudaRuntime, Object[] args) throws InteropException {
            checkArgumentLength(args, 3);
            long destPointer = expectLong(args[0]);
            long fromPointer = expectLong(args[1]);
            long numBytesToCopy = expectPositiveLong(args[2]);
            // cudaMemcpyKind from driver_types.h (default: direction of transfer is
            // inferred from the pointer values, uses virtual addressing)
            final long cudaMemcpyDefault = 4;
            callSymbol(cudaRuntime, destPointer, fromPointer, numBytesToCopy, cudaMemcpyDefault);
            return NoneValue.get();
        }
    };

    private final String name;
    private final String nfiSignature;

    CUDARuntimeFunction(String name, String nfiSignature) {
        this.name = name;
        this.nfiSignature = nfiSignature;
    }

    public String getName() {
        return name;
    }

    public Object getSymbol(CUDARuntime runtime) throws UnknownIdentifierException {
        return runtime.getSymbol(CUDARuntime.CUDA_RUNTIME_LIBRARY_NAME, name, nfiSignature);
    }

    protected void callSymbol(CUDARuntime runtime, Object... arguments) throws UnsupportedTypeException, ArityException, UnsupportedMessageException, UnknownIdentifierException {
        CompilerAsserts.neverPartOfCompilation();
        Object result = CUDARuntime.INTEROP.execute(getSymbol(runtime), arguments);
        runtime.checkCUDAReturnCode(result, getName());
    }
}
