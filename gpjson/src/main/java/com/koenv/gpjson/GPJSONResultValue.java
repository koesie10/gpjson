package com.koenv.gpjson;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.*;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

@ExportLibrary(InteropLibrary.class)
public class GPJSONResultValue implements TruffleObject {
    private final long[] value;

    public GPJSONResultValue(long[] value) {
        this.value = value;
    }

    @ExportMessage
    public boolean hasArrayElements() {
        return true;
    }

    @ExportMessage
    @CompilerDirectives.TruffleBoundary
    public Object readArrayElement(long index) throws UnsupportedMessageException, InvalidArrayIndexException {
        if (index >= value.length) {
            throw InvalidArrayIndexException.create(index);
        }

        return value[(int) index];
    }

    @ExportMessage
    @CompilerDirectives.TruffleBoundary
    public boolean isArrayElementReadable(long index) {
        return index < value.length;
    }

    @ExportMessage
    public long getArraySize() {
        return value.length;
    }
}
