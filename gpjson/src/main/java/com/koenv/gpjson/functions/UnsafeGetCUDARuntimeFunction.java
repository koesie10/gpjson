package com.koenv.gpjson.functions;

import com.koenv.gpjson.debug.CUDARuntimeWrapper;
import com.koenv.gpjson.gpu.CUDARuntime;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

public class UnsafeGetCUDARuntimeFunction extends Function {
    private final CUDARuntime runtime;

    public UnsafeGetCUDARuntimeFunction(CUDARuntime runtime) {
        super("UNSAFE_getCUDARuntime");

        this.runtime = runtime;
    }

    @Override
    @CompilerDirectives.TruffleBoundary
    public Object call(Object[] arguments) throws UnsupportedTypeException, ArityException {
        CUDARuntimeWrapper wrapper = (CUDARuntimeWrapper) arguments[0];

        wrapper.setCudaRuntime(runtime);

        return true;
    }
}
