package com.koenv.gpjson.result;

import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

@ExportLibrary(InteropLibrary.class)
public class NullValue implements TruffleObject {
    public static final NullValue INSTANCE = new NullValue();

    @ExportMessage
    public boolean isNull() {
        return true;
    }

    @Override
    public String toString() {
        return "null";
    }
}
