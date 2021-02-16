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
import com.oracle.truffle.api.CompilerDirectives;

final class CUModule implements AutoCloseable {
    private final CUDARuntime runtime;

    private final String cubinFile;
    /** Pointer to the native CUmodule object. */
    private final long modulePointer;

    boolean closed = false;

    CUModule(CUDARuntime runtime, String cubinFile, long modulePointer) {
        this.runtime = runtime;
        this.cubinFile = cubinFile;
        this.modulePointer = modulePointer;
        this.closed = false;
    }

    public long getModulePointer() {
        if (closed) {
            CompilerDirectives.transferToInterpreter();
            throw new GPJSONException(String.format("cannot get module pointer, module (%016x) already closed", modulePointer));
        }
        return modulePointer;
    }

    public boolean isClosed() {
        return closed;
    }

    @Override
    public boolean equals(Object other) {
        if (other instanceof CUModule) {
            CUModule otherModule = (CUModule) other;
            return otherModule.cubinFile.equals(cubinFile) && otherModule.closed == closed;
        } else {
            return false;
        }
    }

    @Override
    public int hashCode() {
        return cubinFile.hashCode();
    }

    @Override
    public void close() {
        if (!closed) {
            runtime.cuModuleUnload(this);
            closed = true;
        }
    }
}
