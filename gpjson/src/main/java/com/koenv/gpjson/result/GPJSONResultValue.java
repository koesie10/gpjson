package com.koenv.gpjson.result;

import com.koenv.gpjson.functions.Function;
import com.koenv.gpjson.truffle.MemberSet;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.*;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

import java.nio.file.Path;
import java.util.TreeMap;

@ExportLibrary(InteropLibrary.class)
public class GPJSONResultValue implements TruffleObject {
    private final TreeMap<String, Object> memberMap = new TreeMap<>();

    private final Path filePath;
    private final long numberOfElements;
    private final long[] value;

    public GPJSONResultValue(Path filePath, long numberOfElements, long[] value) {
        this.filePath = filePath;
        this.numberOfElements = numberOfElements;
        this.value = value;

        registerFunction(new GPJSONResultToStringFunction(this));
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

        return value[(int) index * 2];
    }

    @ExportMessage
    @CompilerDirectives.TruffleBoundary
    public boolean isArrayElementReadable(long index) {
        return index * 2 < value.length;
    }

    @ExportMessage
    public long getArraySize() {
        return numberOfElements;
    }

    @ExportMessage
    public boolean hasMembers() {
        return true;
    }

    @ExportMessage
    @CompilerDirectives.TruffleBoundary
    public Object getMembers(@SuppressWarnings("unused") boolean includeInternal) {
        return new MemberSet(memberMap.keySet().toArray(new String[0]));
    }

    @ExportMessage
    @CompilerDirectives.TruffleBoundary
    public boolean isMemberReadable(String member) {
        return memberMap.containsKey(member);
    }

    @ExportMessage
    @CompilerDirectives.TruffleBoundary
    public Object readMember(String member) throws UnknownIdentifierException {
        Object entry = memberMap.get(member);
        if (entry == null) {
            throw UnknownIdentifierException.create(member);
        }
        return entry;
    }

    @ExportMessage
    @CompilerDirectives.TruffleBoundary
    public boolean isMemberInvocable(String member) {
        return memberMap.get(member) instanceof Function;
    }

    @ExportMessage
    public Object invokeMember(String member, Object[] arguments, @CachedLibrary(limit = "2") InteropLibrary callLibrary)
            throws UnsupportedMessageException, ArityException, UnknownIdentifierException, UnsupportedTypeException {
        return callLibrary.execute(readMember(member), arguments);
    }

    private void registerFunction(Function function) {
        this.memberMap.put(function.getName(), function);
    }
}
