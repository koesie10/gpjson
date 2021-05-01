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
public class Result implements TruffleObject {
    private final Path filePath;
    private final long numberOfQueries;
    private final long numberOfLines;
    private final long[][] value;
    private final MappedByteBuffer file;

    public Result(Path filePath, long numberOfQueries, long numberOfLines, long[][] value, MappedByteBuffer file) {
        this.filePath = filePath;
        this.numberOfQueries = numberOfQueries;
        this.numberOfLines = numberOfLines;
        this.value = value;
        this.file = file;
    }

    public Path getFilePath() {
        return filePath;
    }

    public long getNumberOfLines() {
        return numberOfLines;
    }

    public long[][] getValue() {
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
        if (index >= numberOfQueries) {
            throw InvalidArrayIndexException.create(index);
        }

        return new ResultArray(numberOfLines, value[(int) index], file);
    }

    @ExportMessage
    @CompilerDirectives.TruffleBoundary
    public boolean isArrayElementReadable(long index) {
        return index < numberOfQueries;
    }

    @ExportMessage
    public long getArraySize() {
        return numberOfQueries;
    }
}
