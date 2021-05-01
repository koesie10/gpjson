package com.koenv.gpjson.result;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

import java.nio.MappedByteBuffer;
import java.nio.file.Path;

@ExportLibrary(InteropLibrary.class)
public class ResultArray implements TruffleObject {
    private final Path filePath;
    private final long numberOfElements;
    private final long[] value;
    private final MappedByteBuffer file;

    public ResultArray(Path filePath, long numberOfElements, long[] value, MappedByteBuffer file) {
        this.filePath = filePath;
        this.numberOfElements = numberOfElements;
        this.value = value;
        this.file = file;
    }

    public Path getFilePath() {
        return filePath;
    }

    public long getNumberOfElements() {
        return numberOfElements;
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
        if (index >= numberOfElements) {
            throw InvalidArrayIndexException.create(index);
        }

        long valueStart = index * 2;

        if (value[(int) valueStart] == -1) {
            return new ResultItem(this, valueStart, 0);
        }

        return new ResultItem(this, valueStart, 1);
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
