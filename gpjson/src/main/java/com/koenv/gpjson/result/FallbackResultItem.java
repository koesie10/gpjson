package com.koenv.gpjson.result;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

import java.util.List;

@ExportLibrary(InteropLibrary.class)
public class FallbackResultItem implements TruffleObject {
    private final List<String> value;

    public FallbackResultItem(List<String> value) {
        this.value = value;
    }

    @ExportMessage
    public boolean hasArrayElements() {
        return true;
    }

    @ExportMessage
    @CompilerDirectives.TruffleBoundary
    public Object readArrayElement(long index) throws UnsupportedMessageException, InvalidArrayIndexException {
        if (index >= value.size()) {
            throw InvalidArrayIndexException.create(index);
        }

        return value.get((int) index);
    }

    @ExportMessage
    @CompilerDirectives.TruffleBoundary
    public boolean isArrayElementReadable(long index) {
        return index < value.size();
    }

    @ExportMessage
    public long getArraySize() {
        return value.size();
    }
}
