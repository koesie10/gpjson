package com.koenv.gpjson.kernel;

public enum GPJSONKernel {
    COUNT_NEWLINES("count_newlines", "file: inout pointer char, n: sint64, result: inout pointer sint32", "com/koenv/gpjson/kernels/count_newlines.cu"),
    CREATE_NEWLINE_INDEX("create_newline_index", "file: inout pointer char, n: sint64, indices: inout pointer sint32, result: inout pointer sint64", "com/koenv/gpjson/kernels/create_newline_index.cu"),
    DISCOVER_STRUCTURE("discover_structure", "file: inout pointer char, n: sint64", "com/koenv/gpjson/kernels/discover_structure.cu"),
    CREATE_ESCAPE_CARRY_INDEX("create_escape_carry_index", "file: inout pointer char, n: sint64, escape_carry_index: inout pointer char", "com/koenv/gpjson/kernels/create_escape_carry_index.cu"),
    CREATE_ESCAPE_INDEX("create_escape_index", "file: inout pointer char, n: sint64, escape_carry_index: inout pointer char, escape_index: inout pointer sint64, escape_index_size: sint64", "com/koenv/gpjson/kernels/create_escape_index.cu"),
    CREATE_QUOTE_INDEX("create_quote_index", "file: inout pointer char, n: sint64, escape_index: inout pointer sint64, quote_index: inout pointer sint64, quote_carry_index: inout pointer char, quote_index_size: sint64", "com/koenv/gpjson/kernels/create_quote_index.cu"),
    CREATE_STRING_INDEX("create_string_index", "n: sint64, quote_index: inout pointer sint64, quote_counts: inout pointer char", "com/koenv/gpjson/kernels/create_string_index.cu");

    private final String name;
    private final String parameterSignature;
    private final String filename;

    GPJSONKernel(String name, String parameterSignature, String filename) {
        this.name = name;
        this.parameterSignature = parameterSignature;
        this.filename = filename;
    }

    public String getName() {
        return name;
    }

    public String getParameterSignature() {
        return parameterSignature;
    }

    public String getFilename() {
        return filename;
    }
}
