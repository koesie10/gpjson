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
public class FallbackResult implements TruffleObject {
    private final List<List<List<String>>> jsonValues;

    public FallbackResult(List<List<List<String>>> jsonValues) {
        this.jsonValues = jsonValues;
    }

    @ExportMessage
    public boolean hasArrayElements() {
        return true;
    }

    @ExportMessage
    @CompilerDirectives.TruffleBoundary
    public Object readArrayElement(long index) throws UnsupportedMessageException, InvalidArrayIndexException {
        if (index >= jsonValues.size()) {
            throw InvalidArrayIndexException.create(index);
        }

        return new FallbackResultArray(jsonValues.get((int) index));
    }

    @ExportMessage
    @CompilerDirectives.TruffleBoundary
    public boolean isArrayElementReadable(long index) {
        return index < jsonValues.size();
    }

    @ExportMessage
    public long getArraySize() {
        return jsonValues.size();
    }
}
