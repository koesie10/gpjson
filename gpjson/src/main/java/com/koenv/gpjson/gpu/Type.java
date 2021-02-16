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
package com.koenv.gpjson.gpu;

import com.koenv.gpjson.GPJSONException;
import com.oracle.truffle.api.CompilerDirectives;

public enum Type {
    BOOLEAN("bool", "uint8", 1, true),
    CHAR("char", "sint8", 1, true),
    SINT8("sint8", "sint8", 1, true),
    UINT8("uint8", "uint8", 1, true),
    CHAR8("char8", "uint8", 1, true),
    CHAR16("char16", "uint16", 2, true),
    SINT16("sint16", "sint16", 2, true),
    UINT16("uint16", "uint16", 2, true),
    CHAR32("char32", "uint32", 4, true),
    SINT32("sint32", "sint32", 4, true),
    UINT32("uint32", "uint32", 4, true),
    WCHAR("wchar", "sint32", 4, true),
    SINT64("sint64", "sint64", 8, true),
    UINT64("uint64", "uint64", 8, true),
    SLL64("sll64", "sint64", 8, true),
    ULL64("ull64", "uint64", 8, true),
    FLOAT("float", "float", 4, true),
    DOUBLE("double", "double", 8, true),
    NFI_POINTER("void", "pointer", 8, false),  // void* (w/o type) as used in NFI
    STRING("string", "string", 8, false),  // const char*
    VOID("void", "void", 0, false);

    private final String nidlTypeName;
    private final String nfiTypeName;
    private final int sizeBytes;
    private final boolean isElementType;

    Type(String nidlTypeName, String nfiTypeName, int sizeBytes, boolean isElementType) {
        this.nidlTypeName = nidlTypeName;
        this.nfiTypeName = nfiTypeName;
        this.sizeBytes = sizeBytes;
        this.isElementType = isElementType;
    }

    public int getSizeBytes() {
        return this.sizeBytes;
    }

    public boolean isElementType() {
        return this.isElementType;
    }

    public String getNIDLTypeName() {
        return this.nidlTypeName;
    }

    public String getNFITypeName() {
        return this.nfiTypeName;
    }

    public static Type fromNIDLTypeString(String type) throws GPJSONException {
        switch (type) {
            case "bool":
                return Type.BOOLEAN;
            case "char":
                return Type.CHAR;
            case "sint8":
                return Type.SINT8;
            case "uint8":
                return Type.UINT8;
            case "char16":
                return Type.CHAR16;
            case "sint16":
                return Type.SINT16;
            case "uint16":
                return Type.UINT16;
            case "char32":
                return Type.CHAR32;
            case "sint32":
                return Type.SINT32;
            case "uint32":
                return Type.UINT32;
            case "wchar":
                return Type.WCHAR;
            case "sint64":
                return Type.SINT64;
            case "uint64":
                return Type.UINT64;
            case "sll64":
                return Type.SLL64;
            case "ull64":
                return Type.ULL64;
            case "float":
                return Type.FLOAT;
            case "double":
                return Type.DOUBLE;
            case "pointer":
                return Type.NFI_POINTER;
            case "string":
                return Type.STRING;
            case "void":
                return Type.VOID;
            default:
                CompilerDirectives.transferToInterpreter();
                throw new GPJSONException("invalid type '" + type + "'");
        }
    }
}
