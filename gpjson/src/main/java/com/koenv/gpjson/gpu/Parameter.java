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

import java.util.ArrayList;

import com.koenv.gpjson.GPJSONException;
import com.oracle.truffle.api.CompilerAsserts;

public class Parameter {

    public enum Kind {
        BY_VALUE,
        POINTER_IN,
        POINTER_OUT,
        POINTER_INOUT,
    }

    private int position;
    private final String name;
    private final Type type;
    private final Kind kind;

    /**
     * Create new parameter from its components.
     *
     * @param position zero-based position from the left of the parameter list
     * @param name parameter name
     * @param type data type of the parameter
     * @param kind kind of the parameter (by-value, pointer with direction)
     */
    Parameter(int position, String name, Type type, Kind kind) {
        this.position = position;
        this.name = name;
        this.type = type;
        this.kind = kind;
    }

    /**
     * Create new pointer parameter from its components (with position 0).
     *
     * @param name parameter name
     * @param type data type of the value to which the pointer points
     * @param kind direction of pointer parameter (allowed values `POINTER_IN`, `POINTER_OUT` and
     *            `POINTER_INOUT`, must not by `BY_VALUE`)
     */
    public static Parameter createPointerParameter(String name, Type type, Kind kind) {
        assert kind != Kind.BY_VALUE : "pointer parameter cannot be by-value";
        return new Parameter(0, name, type, kind);
    }

    /**
     * Create new by-value parameter from its components (with position 0).
     *
     * @param name parameter name
     * @param type data type of the parameter
     */
    public static Parameter createByValueParameter(String name, Type type) {
        return new Parameter(0, name, type, Kind.BY_VALUE);
    }

    /**
     * Parse parameter string in NIDL or legacy Truffle NFI syntax.
     *
     * <pre>
     * NIDL
     *  paramStr ::= parameterName ":" [direction "pointer"] NIDLTypeName
     *  direction ::= "in" | "out" | "inout"
     *
     * NFI
     *  paramStr :: = NFITypeName
     * </pre>
     *
     * @param param string to be parsed in NIDL or legacy NFI syntax
     * @throws GPJSONException if {@code param} string cannot be parsed successfully
     * @return Parameter
     */
    private static Parameter parseNIDLParameterString(String param) throws GPJSONException {
        String paramStr = param.trim();
        String[] nameAndType = paramStr.split(":");
        if (nameAndType.length != 2) {
            throw new GPJSONException("expected parameter as \"name: type\", got " + paramStr);
        }
        String name = nameAndType[0].trim();
        String extTypeStr = nameAndType[1].trim();
        String[] dirPointerAndType = extTypeStr.split("(\\s)+");
        if (dirPointerAndType.length != 1 && dirPointerAndType.length != 3) {
            throw new GPJSONException("expected type, got " + extTypeStr);
        }
        if (dirPointerAndType.length == 1) {
            Type type = Type.fromNIDLTypeString(dirPointerAndType[0]);
            if (type == Type.NFI_POINTER) {
                // the NFI pointer is not a legal by-value parameter type
                throw new GPJSONException("invalid type \"pointer\" of by-value parameter");
            }
            if (type == Type.VOID) {
                // the void is not a legal by-value parameter type
                throw new GPJSONException("invalid type \"pointer\" of by-value parameter");
            }
            return createByValueParameter(name, type);
        } else {
            if (dirPointerAndType[1].equals("pointer")) {
                Type type = Type.fromNIDLTypeString(dirPointerAndType[2]);
                if (type == Type.NFI_POINTER) {
                    // the NFI pointer may not appear as this NIDL pointer's type
                    throw new GPJSONException("invalid type \"pointer\"");
                }
                switch (dirPointerAndType[0]) {
                    case "in":
                        return createPointerParameter(name, type, Kind.POINTER_IN);
                    case "inout":
                        return createPointerParameter(name, type, Kind.POINTER_INOUT);
                    case "out":
                        return createPointerParameter(name, type, Kind.POINTER_OUT);
                    default:
                        throw new GPJSONException("invalid direction: " + dirPointerAndType[0] + ", expected \"in\", \"inout\", or \"out\"");
                }
            } else {
                throw new GPJSONException("expected keyword \"pointer\"");
            }
        }
    }

    public static ArrayList<Parameter> parseParameterSignature(String parameterSignature) throws GPJSONException {
        CompilerAsserts.neverPartOfCompilation();
        ArrayList<Parameter> params = new ArrayList<>();
        for (String s : parameterSignature.trim().split(",")) {
            params.add(parseNIDLParameterString(s.trim()));
        }
        return params;
    }

    public Type getType() {
        return type;
    }

    public boolean isPointer() {
        return !(kind == Kind.BY_VALUE);
    }

    public String getName() {
        return name;
    }

    public Kind getKind() {
        return kind;
    }

    public void setPosition(int position) {
        this.position = position;
    }

    public int getPosition() {
        return position;
    }

    @Override
    public String toString() {
        return "Parameter(position=" + position + ", name=" + name + ", type=" + type + ", kind=" + kind + ")";
    }
}
