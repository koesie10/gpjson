package com.koenv.gpjson;

import com.koenv.gpjson.functions.Function;
import com.koenv.gpjson.functions.QueryFunction;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.*;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

import java.util.Arrays;
import java.util.TreeMap;

@ExportLibrary(InteropLibrary.class)
public class GPJSONLibrary implements TruffleObject {
    private final TreeMap<String, Object> map = new TreeMap<>();

    public GPJSONLibrary() {
        this.map.put("query", new QueryFunction());
    }

    @ExportMessage
    public boolean hasMembers() {
        return true;
    }

    @ExportMessage
    @CompilerDirectives.TruffleBoundary
    public Object getMembers(@SuppressWarnings("unused") boolean includeInternal) {
        return new MemberSet(map.keySet().toArray(new String[0]));
    }

    @ExportMessage
    @CompilerDirectives.TruffleBoundary
    public boolean isMemberReadable(String member) {
        return map.containsKey(member);
    }

    @ExportMessage
    @CompilerDirectives.TruffleBoundary
    public Object readMember(String member) throws UnknownIdentifierException {
        Object entry = map.get(member);
        if (entry == null) {
            throw UnknownIdentifierException.create(member);
        }
        return entry;
    }

    @ExportMessage
    @CompilerDirectives.TruffleBoundary
    public boolean isMemberInvocable(String member) {
        return map.get(member) instanceof Function;
    }

    @ExportMessage
    public Object invokeMember(String member, Object[] arguments, @CachedLibrary(limit = "2") InteropLibrary callLibrary)
            throws UnsupportedMessageException, ArityException, UnknownIdentifierException, UnsupportedTypeException {
        return callLibrary.execute(readMember(member), arguments);
    }

    @ExportLibrary(InteropLibrary.class)
    public static final class MemberSet implements TruffleObject {

        @CompilerDirectives.CompilationFinal(dimensions = 1)
        private final String[] values;

        public MemberSet(String... values) {
            this.values = values;
        }

        @ExportMessage
        @SuppressWarnings("static-method")
        public boolean hasArrayElements() {
            return true;
        }

        @ExportMessage
        public long getArraySize() {
            return values.length;
        }

        @ExportMessage
        public boolean isArrayElementReadable(long index) {
            return index >= 0 && index < values.length;
        }

        @ExportMessage
        public Object readArrayElement(long index) throws InvalidArrayIndexException {
            if ((index < 0) || (index >= values.length)) {
                CompilerDirectives.transferToInterpreter();
                throw InvalidArrayIndexException.create(index);
            }
            return values[(int) index];
        }

        @CompilerDirectives.TruffleBoundary
        public boolean containsValue(String name) {
            return Arrays.asList(values).contains(name);
        }
    }
}
