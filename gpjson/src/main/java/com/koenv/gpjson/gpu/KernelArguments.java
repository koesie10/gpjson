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

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

final class KernelArguments implements Closeable {

    private final UnsafeHelper.PointerArray argumentArray;
    private final List<Closeable> argumentValues = new ArrayList<>();

    public KernelArguments(int numArgs) {
        this.argumentArray = UnsafeHelper.createPointerArray(numArgs);
    }

    public KernelArguments(List<UnsafeHelper.MemoryObject> arguments) {
        this.argumentArray = UnsafeHelper.createPointerArray(arguments.size());
        for (int i = 0; i < arguments.size(); i++) {
            setArgument(i, arguments.get(i));
        }
    }

    public void setArgument(int argIdx, UnsafeHelper.MemoryObject obj) {
        argumentArray.setValueAt(argIdx, obj.getAddress());
        argumentValues.add(obj);
    }

    long getPointer() {
        return argumentArray.getAddress();
    }

    @Override
    public void close() {
        this.argumentArray.close();
        for (Closeable c : argumentValues) {
            try {
                c.close();
            } catch (IOException e) {
                /* ignored */
            }
        }
    }
}
