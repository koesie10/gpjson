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
package com.koenv.gpjson;

import com.koenv.gpjson.gpu.CUDARuntime;
import com.oracle.truffle.api.TruffleLanguage;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class GPJSONContext {
    private final TruffleLanguage.Env env;
    private final CUDARuntime cudaRuntime;
    private final GPJSONLibrary root;

    private volatile boolean cudaInitialized = false;
    private AtomicInteger moduleId = new AtomicInteger(0);

    private final List<Runnable> disposables = new ArrayList<>();

    public GPJSONContext(TruffleLanguage.Env env) {
        this.env = env;

        this.cudaRuntime = new CUDARuntime(this, env);

        this.root = new GPJSONLibrary(this);
    }

    public TruffleLanguage.Env getEnv() {
        return env;
    }

    public CUDARuntime getCudaRuntime() {
        return cudaRuntime;
    }

    public GPJSONLibrary getRoot() {
        return root;
    }

    public void addDisposable(Runnable disposable) {
        disposables.add(disposable);
    }

    public void dispose() {
        for (Runnable runnable : disposables) {
            runnable.run();
        }
    }

    public boolean isCUDAInitialized() {
        return cudaInitialized;
    }

    public void setCUDAInitialized() {
        cudaInitialized = true;
    }

    public int getNextModuleId() {
        return moduleId.incrementAndGet();
    }
}
