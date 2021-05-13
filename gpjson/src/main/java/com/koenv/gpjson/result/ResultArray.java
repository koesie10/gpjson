package com.koenv.gpjson.result;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

import java.nio.MappedByteBuffer;

@ExportLibrary(InteropLibrary.class)
public class ResultArray implements TruffleObject {
    private final long numberOfLines;
    private final int resultsPerLine;
    private final long[] value;
    private final MappedByteBuffer file;

    public ResultArray(long numberOfLines, int resultsPerLine, long[] value, MappedByteBuffer file) {
        this.numberOfLines = numberOfLines;
        this.resultsPerLine = resultsPerLine;
        this.value = value;
        this.file = file;
    }

    public long getNumberOfLines() {
        return numberOfLines;
    }

    public int getResultsPerLine() {
        return resultsPerLine;
    }

    public long[] getValue() {
        return value;
    }

    public MappedByteBuffer getFile() {
        return file;
    }

    @ExportMessage
    public boolean hasArrayElements() {
        return true;
    }

    @ExportMessage
    @CompilerDirectives.TruffleBoundary
    public Object readArrayElement(long index) throws UnsupportedMessageException, InvalidArrayIndexException {
        if (index >= numberOfLines) {
            throw InvalidArrayIndexException.create(index);
        }

        long valueStart = index * 2 * resultsPerLine;

        return new ResultItem(this, valueStart, resultsPerLine);
    }

    @ExportMessage
    @CompilerDirectives.TruffleBoundary
    public boolean isArrayElementReadable(long index) {
        return index < numberOfLines;
    }

    @ExportMessage
    public long getArraySize() {
        return numberOfLines;
    }
}
