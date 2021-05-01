package com.koenv.gpjson.result;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

import java.nio.charset.StandardCharsets;

@ExportLibrary(InteropLibrary.class)
public class ResultItem implements TruffleObject {
    private final ResultArray array;
    private final long startIndex;
    private final long numberOfElements;

    public ResultItem(ResultArray array, long startIndex, long numberOfElements) {
        this.array = array;
        this.startIndex = startIndex;
        this.numberOfElements = numberOfElements;
    }

    @ExportMessage
    public boolean hasArrayElements() {
        return true;
    }

    @ExportMessage
    @CompilerDirectives.TruffleBoundary
    public Object readArrayElement(long index) throws UnsupportedMessageException, InvalidArrayIndexException {
        if (index >= numberOfElements) {
            throw InvalidArrayIndexException.create(index);
        }

        long valueIndex = startIndex + index * 2;

        int valueStart = (int) array.getValue()[(int) valueIndex];

        if (valueStart == -1) {
            return NullValue.INSTANCE;
        }

        int valueEnd = (int) array.getValue()[(int) (valueIndex + 1)];

        byte[] value = new byte[valueEnd - valueStart];
        array.getFile().position(valueStart);
        array.getFile().get(value);

        return new String(value, StandardCharsets.UTF_8);
    }

    @ExportMessage
    @CompilerDirectives.TruffleBoundary
    public boolean isArrayElementReadable(long index) {
        return index < numberOfElements;
    }

    @ExportMessage
    public long getArraySize() {
        return numberOfElements;
    }
}
