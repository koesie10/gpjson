package com.koenv.gpjson.functions;

import com.koenv.gpjson.debug.Timings;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

public class ExportTimingsFunction extends Function {

    public ExportTimingsFunction() {
        super("exportTimings");
    }

    @Override
    @CompilerDirectives.TruffleBoundary
    public Object call(Object[] arguments) throws UnsupportedTypeException, ArityException {
        checkArgumentLength(arguments, 0);

        return Timings.TIMINGS.export();
    }
}
