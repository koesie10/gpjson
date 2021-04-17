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

import com.koenv.gpjson.GPJSONContext;
import com.koenv.gpjson.GPJSONException;
import com.koenv.gpjson.debug.Timings;
import com.koenv.gpjson.kernel.GPJSONKernel;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.TruffleLanguage;
import com.oracle.truffle.api.exception.AbstractTruffleException;
import com.oracle.truffle.api.interop.*;
import com.oracle.truffle.api.source.Source;
import org.graalvm.collections.Pair;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

public class CUDARuntime {
    public static final String CUDA_RUNTIME_LIBRARY_NAME = "cudart";
    public static final String CUDA_LIBRARY_NAME = "cuda";
    static final String NVRTC_LIBRARY_NAME = "nvrtc";

    /**
     * Map from library-path to NFI library.
     */
    private final Map<String, TruffleObject> loadedLibraries = new HashMap<>();

    /**
     * Map of (library-path, symbol-name) to callable.
     */
    private final Map<Pair<String, String>, Object> boundFunctions = new HashMap<>();

    private Map<GPJSONKernel, Kernel> loadedKernels = new HashMap<>();

    // using this slow/uncached instance since all calls are non-critical
    public static final InteropLibrary INTEROP = InteropLibrary.getFactory().getUncached();

    private final GPJSONContext context;
    private final NVRuntimeCompiler nvrtc;

    public final Timings timings = Timings.TIMINGS;

    public CUDARuntime(GPJSONContext context, TruffleLanguage.Env env) {
        this.context = context;
        try {
            TruffleObject libcudart = (TruffleObject) env.parseInternal(
                    Source.newBuilder("nfi", "load " + "lib" + CUDA_RUNTIME_LIBRARY_NAME + ".so", "cudaruntime").build()).call();
            TruffleObject libcuda = (TruffleObject) env.parseInternal(
                    Source.newBuilder("nfi", "load " + "lib" + CUDA_LIBRARY_NAME + ".so", "cuda").build()).call();
            TruffleObject libnvrtc = (TruffleObject) env.parseInternal(
                    Source.newBuilder("nfi", "load " + "lib" + NVRTC_LIBRARY_NAME + ".so", "nvrtc").build()).call();
            loadedLibraries.put(CUDA_RUNTIME_LIBRARY_NAME, libcudart);
            loadedLibraries.put(CUDA_LIBRARY_NAME, libcuda);
            loadedLibraries.put(NVRTC_LIBRARY_NAME, libnvrtc);
        } catch (UnsatisfiedLinkError e) {
            throw new GPJSONException(e.getMessage());
        }

        nvrtc = new NVRuntimeCompiler(this);
        context.addDisposable(this::shutdown);
    }

    private void shutdown() {
        // unload all modules
        for (Kernel kernel : loadedKernels.values()) {
            try {
                kernel.getModule().close();
            } catch (Exception e) {
                /* ignore exception */
            }
        }
        loadedKernels.clear();
    }

    public ManagedGPUMemory allocateMemory(long numBytes) {
        return new ManagedGPUMemory(this, cudaMallocManaged(numBytes));
    }

    public ManagedGPUPointer allocateUnmanagedMemory(long numBytes) {
        return new ManagedGPUPointer(this, cudaMalloc(numBytes), numBytes, numBytes, null);
    }

    public ManagedGPUPointer allocateUnmanagedMemory(long numElements, Type type) {
        long numBytes = numElements * type.getSizeBytes();
        return new ManagedGPUPointer(this, cudaMalloc(numBytes), numBytes, numElements, type);
    }

    @CompilerDirectives.TruffleBoundary
    public GPUPointer cudaMalloc(long numBytes) {
        timings.start("cudaMalloc");
        try (UnsafeHelper.PointerObject outPointer = UnsafeHelper.createPointerObject()) {
            Object callable = CUDARuntimeFunction.CUDA_MALLOC.getSymbol(this);
            Object result = INTEROP.execute(callable, outPointer.getAddress(), numBytes);
            checkCUDAReturnCode(result, "cudaMalloc");
            long addressAllocatedMemory = outPointer.getValueOfPointer();
            return new GPUPointer(addressAllocatedMemory);
        } catch (InteropException e) {
            throw new GPJSONException(e);
        } finally {
            timings.end();
        }
    }

    @CompilerDirectives.TruffleBoundary
    public LittleEndianNativeArrayView cudaMallocManaged(long numBytes) {
        timings.start("cudaMallocManaged");
        final int cudaMemAttachGlobal = 0x01;
        try (UnsafeHelper.PointerObject outPointer = UnsafeHelper.createPointerObject()) {
            Object callable = CUDARuntimeFunction.CUDA_MALLOCMANAGED.getSymbol(this);
            Object result = INTEROP.execute(callable, outPointer.getAddress(), numBytes, cudaMemAttachGlobal);
            checkCUDAReturnCode(result, "cudaMallocManaged");
            long addressAllocatedMemory = outPointer.getValueOfPointer();
            return new LittleEndianNativeArrayView(addressAllocatedMemory, numBytes);
        } catch (InteropException e) {
            throw new GPJSONException(e);
        } finally {
            timings.end();
        }
    }

    @CompilerDirectives.TruffleBoundary
    public void cudaFree(LittleEndianNativeArrayView memory) {
        timings.start("cudaFree");

        try {
            Object callable = CUDARuntimeFunction.CUDA_FREE.getSymbol(this);
            Object result = INTEROP.execute(callable, memory.getStartAddress());
            checkCUDAReturnCode(result, "cudaFree");
        } catch (InteropException e) {
            throw new GPJSONException(e);
        } finally {
            timings.end();
        }
    }

    @CompilerDirectives.TruffleBoundary
    public void cudaFree(GPUPointer pointer) {
        timings.start("cudaFree");

        try {
            Object callable = CUDARuntimeFunction.CUDA_FREE.getSymbol(this);
            Object result = INTEROP.execute(callable, pointer.getRawPointer());
            checkCUDAReturnCode(result, "cudaFree");
        } catch (InteropException e) {
            throw new GPJSONException(e);
        } finally {
            timings.end();
        }
    }

    @CompilerDirectives.TruffleBoundary
    public void cudaDeviceSynchronize() {
        timings.start("cudaDeviceSynchronize");
        try {
            Object callable = CUDARuntimeFunction.CUDA_DEVICESYNCHRONIZE.getSymbol(this);
            Object result = INTEROP.execute(callable);
            checkCUDAReturnCode(result, "cudaDeviceSynchronize");
        } catch (InteropException e) {
            throw new GPJSONException(e);
        } finally {
            timings.end();
        }
    }

    @CompilerDirectives.TruffleBoundary
    public String cudaGetErrorString(int errorCode) {
        try {
            Object callable = CUDARuntimeFunction.CUDA_GETERRORSTRING.getSymbol(this);
            Object result = INTEROP.execute(callable, errorCode);
            return INTEROP.asString(result);
        } catch (InteropException e) {
            throw new GPJSONException(e);
        }
    }

    @CompilerDirectives.TruffleBoundary
    public void cudaMemcpy(long destPointer, long fromPointer, long numBytesToCopy, CUDAMemcpyKind kind) {
        timings.start("cudaMemcpy");
        try {
            Object callable = CUDARuntimeFunction.CUDA_MEMCPY.getSymbol(this);
            if (numBytesToCopy < 0) {
                throw new IllegalArgumentException("requested negative number of bytes to copy " + numBytesToCopy);
            }
            Object result = INTEROP.execute(callable, destPointer, fromPointer, numBytesToCopy, kind.getKind());
            checkCUDAReturnCode(result, "cudaMemcpy");
        } catch (InteropException e) {
            throw new GPJSONException(e);
        } finally {
            timings.end();
        }
    }

    public Kernel getKernel(GPJSONKernel kernelType) {
        timings.start("getKernel", kernelType.getName());

        try {
            return loadedKernels.computeIfAbsent(kernelType, this::loadKernel);
        } finally {
            timings.end();
        }
    }

    private Kernel loadKernel(GPJSONKernel kernelType) {
        timings.start("loadKernel", kernelType.getName());
        String code;
        try (InputStream is = getClass().getClassLoader().getResourceAsStream(kernelType.getFilename())) {
            if (is == null) {
                throw new GPJSONException("Missing kernel code file " + kernelType.getFilename());
            }

            code = new BufferedReader(new InputStreamReader(is)).lines().collect(Collectors.joining("\n"));
        } catch (IOException e) {
            throw new GPJSONException("Failed to read kernel code", e, AbstractTruffleException.UNLIMITED_STACK_TRACE, null);
        }

        String moduleName = "truffle" + context.getNextModuleId();
        PTXKernel ptx = nvrtc.compileKernel(code, kernelType.getName(), moduleName, "--std=c++14");
        CUModule module = cuModuleLoadData(ptx.getPtxSource(), moduleName);
        long kernelFunctionHandle = cuModuleGetFunction(module, ptx.getLoweredKernelName());

        Kernel kernel = new Kernel(this, kernelType.getName(), ptx.getLoweredKernelName(), kernelFunctionHandle, kernelType.getParameterSignature(), module, ptx.getPtxSource());
        timings.end();

        return kernel;
    }

    @CompilerDirectives.TruffleBoundary
    public CUModule cuModuleLoad(String cubinName) {
        timings.start("cuModuleLoad", cubinName);
        assertCUDAInitialized();
        try (UnsafeHelper.Integer64Object modulePtr = UnsafeHelper.createInteger64Object()) {
            Object callable = CUDADriverFunction.CU_MODULELOAD.getSymbol(this);
            Object result = INTEROP.execute(callable, modulePtr.getAddress(), cubinName);
            checkCUReturnCode(result, "cuModuleLoad");
            return new CUModule(this, cubinName, modulePtr.getValue());
        } catch (InteropException e) {
            throw new GPJSONException(e);
        } finally {
            timings.end();
        }
    }

    @CompilerDirectives.TruffleBoundary
    public CUModule cuModuleLoadData(String ptx, String moduleName) {
        timings.start("cuModuleLoadData");
        assertCUDAInitialized();
        try (UnsafeHelper.Integer64Object modulePtr = UnsafeHelper.createInteger64Object()) {
            Object callable = CUDADriverFunction.CU_MODULELOADDATA.getSymbol(this);
            Object result = INTEROP.execute(callable, modulePtr.getAddress(), ptx);
            checkCUReturnCode(result, "cuModuleLoadData");
            return new CUModule(this, moduleName, modulePtr.getValue());
        } catch (InteropException e) {
            throw new GPJSONException(e);
        } finally {
            timings.end();
        }
    }

    @CompilerDirectives.TruffleBoundary
    public void cuModuleUnload(CUModule module) {
        try {
            Object callable = CUDADriverFunction.CU_MODULEUNLOAD.getSymbol(this);
            Object result = INTEROP.execute(callable, module.getModulePointer());
            checkCUReturnCode(result, "cuModuleUnload");
        } catch (InteropException e) {
            throw new GPJSONException(e);
        }
    }

    /**
     * Get function handle to kernel in module.
     *
     * @param kernelModule CUmodule containing the kernel function
     * @param kernelName
     * @return native CUfunction function handle
     */
    @CompilerDirectives.TruffleBoundary
    public long cuModuleGetFunction(CUModule kernelModule, String kernelName) {
        timings.start("cuModuleGetFunction");
        try (UnsafeHelper.Integer64Object functionPtr = UnsafeHelper.createInteger64Object()) {
            Object callable = CUDADriverFunction.CU_MODULEGETFUNCTION.getSymbol(this);
            Object result = INTEROP.execute(callable,
                    functionPtr.getAddress(), kernelModule.getModulePointer(), kernelName);
            checkCUReturnCode(result, "cuModuleGetFunction");
            return functionPtr.getValue();
        } catch (InteropException e) {
            throw new GPJSONException(e);
        } finally {
            timings.end();
        }
    }

    @CompilerDirectives.TruffleBoundary
    public void cuCtxSynchronize() {
        assertCUDAInitialized();
        try {
            Object callable = CUDADriverFunction.CU_CTXSYNCHRONIZE.getSymbol(this);
            Object result = INTEROP.execute(callable);
            checkCUReturnCode(result, "cuCtxSynchronize");
        } catch (InteropException e) {
            throw new GPJSONException(e);
        }
    }

    public void cuLaunchKernel(Kernel kernel, Dim3 gridSize, Dim3 blockSize, int dynamicSharedMemoryBytes, int stream, KernelArguments args) {
        try {
            timings.start("cuLaunchKernel", kernel.getKernelName());
            Object callable = CUDADriverFunction.CU_LAUNCHKERNEL.getSymbol(this);
            Object result = INTEROP.execute(callable,
                    kernel.getKernelFunctionHandle(),
                    gridSize.getX(),
                    gridSize.getY(),
                    gridSize.getZ(),
                    blockSize.getX(),
                    blockSize.getY(),
                    blockSize.getZ(),
                    dynamicSharedMemoryBytes,
                    stream,
                    args.getPointer(),              // pointer to kernel arguments array
                    0                               // extra args
            );
            checkCUReturnCode(result, "cuLaunchKernel");
            timings.end();
            cudaDeviceSynchronize();
        } catch (InteropException e) {
            throw new GPJSONException(e);
        }
    }

    @CompilerDirectives.TruffleBoundary
    private void cuInit() {
        try {
            Object callable = CUDADriverFunction.CU_INIT.getSymbol(this);
            int flags = 0; // must be zero as per CUDA Driver API documentation
            Object result = INTEROP.execute(callable, flags);
            checkCUReturnCode(result, "cuInit");
        } catch (InteropException e) {
            throw new GPJSONException(e);
        }
    }

    @CompilerDirectives.TruffleBoundary
    private int cuDeviceGetCount() {
        try (UnsafeHelper.Integer32Object devCount = UnsafeHelper.createInteger32Object()) {
            Object callable = CUDADriverFunction.CU_DEVICEGETCOUNT.getSymbol(this);
            Object result = INTEROP.execute(callable, devCount.getAddress());
            checkCUReturnCode(result, "cuDeviceGetCount");
            return devCount.getValue();
        } catch (InteropException e) {
            throw new GPJSONException(e);
        }
    }

    @CompilerDirectives.TruffleBoundary
    private int cuDeviceGet(int deviceOrdinal) {
        assertCUDAInitialized();
        try (UnsafeHelper.Integer32Object deviceObj = UnsafeHelper.createInteger32Object()) {
            Object callable = CUDADriverFunction.CU_DEVICEGET.getSymbol(this);
            Object result = INTEROP.execute(callable, deviceObj.getAddress(), deviceOrdinal);
            checkCUReturnCode(result, "cuDeviceGet");
            return deviceObj.getValue();
        } catch (InteropException e) {
            throw new GPJSONException(e);
        }
    }

    @CompilerDirectives.TruffleBoundary
    private String cuDeviceGetName(int cuDeviceId) {
        final int maxLength = 256;
        try (UnsafeHelper.StringObject nameString = new UnsafeHelper.StringObject(maxLength)) {
            Object callable = CUDADriverFunction.CU_DEVICEGETNAME.getSymbol(this);
            Object result = INTEROP.execute(callable, nameString.getAddress(), maxLength, cuDeviceId);
            checkCUReturnCode(result, "cuDeviceGetName");
            return nameString.getZeroTerminatedString();
        } catch (InteropException e) {
            throw new GPJSONException(e);
        }
    }

    @CompilerDirectives.TruffleBoundary
    private long cuCtxCreate(int flags, int cudevice) {
        try (UnsafeHelper.PointerObject pctx = UnsafeHelper.createPointerObject()) {
            Object callable = CUDADriverFunction.CU_CTXCREATE.getSymbol(this);
            Object result = INTEROP.execute(callable, pctx.getAddress(), flags, cudevice);
            checkCUReturnCode(result, "cuCtxCreate");
            return pctx.getValueOfPointer();
        } catch (InteropException e) {
            throw new GPJSONException(e);
        }
    }

    @CompilerDirectives.TruffleBoundary
    private long cuDevicePrimaryCtxRetain(int cudevice) {
        try (UnsafeHelper.PointerObject pctx = UnsafeHelper.createPointerObject()) {
            Object callable = CUDADriverFunction.CU_DEVICEPRIMARYCTXRETAIN.getSymbol(this);
            Object result = INTEROP.execute(callable, pctx.getAddress(), cudevice);
            checkCUReturnCode(result, "cuDevicePrimaryCtxRetain");
            return pctx.getValueOfPointer();
        } catch (InteropException e) {
            throw new GPJSONException(e);
        }
    }

    @CompilerDirectives.TruffleBoundary
    private void cuCtxDestroy(long ctx) {
        try {
            Object callable = CUDADriverFunction.CU_CTXCREATE.getSymbol(this);
            Object result = INTEROP.execute(callable, ctx);
            checkCUReturnCode(result, "cuCtxDestroy");
        } catch (InteropException e) {
            throw new GPJSONException(e);
        }
    }

    @CompilerDirectives.TruffleBoundary
    private void assertCUDAInitialized() {
        timings.start("assertCUDAInitialized");
        if (!context.isCUDAInitialized()) {
            // a simple way to create the device context in the driver is to call CUDA function
            cudaDeviceSynchronize();
            context.setCUDAInitialized();
        }
        timings.end();
    }

    @SuppressWarnings("static-method")
    private static void checkCUReturnCode(Object result, String... function) {
        int returnCode;
        try {
            returnCode = INTEROP.asInt(result);
        } catch (UnsupportedMessageException e) {
            CompilerDirectives.transferToInterpreter();
            throw new GPJSONException(
                    "expected return code as Integer object in " + Arrays.toString(function) + ", got " +
                            result.getClass().getName());
        }
        if (returnCode != 0) {
            throw new GPJSONException(returnCode, DriverAPIErrorMessages.getString(returnCode), function);
        }
    }

    public void checkCUDAReturnCode(Object result, String... function) {
        if (!(result instanceof Integer)) {
            CompilerDirectives.transferToInterpreter();
            throw new GPJSONException("expected return code as Integer object in " + GPJSONException.format(function) + ", got " + result.getClass().getName());
        }
        Integer returnCode = (Integer) result;
        if (returnCode != 0) {
            CompilerDirectives.transferToInterpreter();
            throw new GPJSONException(returnCode, cudaGetErrorString(returnCode), function);
        }
    }

    /**
     * Get function as callable from native library.
     *
     * @param libraryPath path to library (.so file)
     * @param symbolName name of the function (symbol) too look up
     * @param nfiSignature NFI signature of the function
     * @return a callable as a TruffleObject
     */
    @CompilerDirectives.TruffleBoundary
    public Object getSymbol(String libraryPath, String symbolName, String nfiSignature) throws UnknownIdentifierException {
        return getSymbol(libraryPath, symbolName, nfiSignature, "");
    }

    /**
     * Get function as callable from native library.
     *
     * @param libraryPath path to library (.so file)
     * @param symbolName name of the function (symbol) too look up
     * @param nfiSignature NFI signature of the function
     * @param hint additional string shown to user when symbol cannot be loaded
     * @return a callable as a TruffleObject
     */
    @CompilerDirectives.TruffleBoundary
    public Object getSymbol(String libraryPath, String symbolName, String nfiSignature, String hint) throws UnknownIdentifierException {
        Pair<String, String> functionKey = Pair.create(libraryPath, symbolName);
        Object callable = boundFunctions.get(functionKey);
        if (callable == null) {
            // symbol does not exist or not yet bound
            TruffleObject library = loadedLibraries.get(libraryPath);
            if (library == null) {
                try {
                    // library does not exist or is not loaded yet
                    library = (TruffleObject) context.getEnv().parseInternal(Source.newBuilder("nfi", "load \"" + libraryPath + "\"", libraryPath).build()).call();
                } catch (UnsatisfiedLinkError e) {
                    throw new GPJSONException("unable to load shared library '" + libraryPath + "': " + e.getMessage() + hint);
                }

                loadedLibraries.put(libraryPath, library);
            }
            try {
                Object symbol = INTEROP.readMember(library, symbolName);
                callable = INTEROP.invokeMember(symbol, "bind", nfiSignature);
            } catch (UnsatisfiedLinkError | UnsupportedMessageException | ArityException | UnsupportedTypeException e) {
                throw new GPJSONException("unexpected behavior: " + e.getMessage());
            }
            boundFunctions.put(functionKey, callable);
        }
        return callable;
    }
}
