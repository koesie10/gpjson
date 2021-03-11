/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, 2020, Oracle and/or its affiliates. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
package com.koenv.gpjson;

import com.koenv.gpjson.functions.Function;
import com.koenv.gpjson.functions.QueryFunction;
import com.koenv.gpjson.functions.QueryGPUFunction;
import com.koenv.gpjson.functions.UnsafeGetCUDARuntimeFunction;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.*;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

import java.util.Arrays;
import java.util.Objects;
import java.util.TreeMap;

@ExportLibrary(InteropLibrary.class)
public class GPJSONLibrary implements TruffleObject {
    private final TreeMap<String, Object> map = new TreeMap<>();

    private final UnsafeGetCUDARuntimeFunction unsafeGetCUDARuntimeFunction;

    public GPJSONLibrary(GPJSONContext context) {
        this.map.put("query", new QueryFunction());
        this.map.put("queryGPU", new QueryGPUFunction(context));

        unsafeGetCUDARuntimeFunction = new UnsafeGetCUDARuntimeFunction(context.getCudaRuntime());
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
        // This is used for retrieving the CUDARuntime in tests
        if (Objects.equals(member, "UNSAFE_getCUDARuntime")) {
            return callLibrary.execute(unsafeGetCUDARuntimeFunction, arguments);
        }

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
