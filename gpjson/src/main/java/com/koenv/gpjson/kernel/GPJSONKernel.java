package com.koenv.gpjson.kernel;

public enum GPJSONKernel {
    COUNT_NEWLINES("count_newlines", "file: inout pointer char, n: sint64, result: inout pointer sint32", "com/koenv/gpjson/kernels/count_newlines.cu"),
    CREATE_NEWLINE_INDEX("create_newline_index", "file: inout pointer char, n: sint64, indices: inout pointer sint32, result: inout pointer sint64", "com/koenv/gpjson/kernels/create_newline_index.cu"),
    CREATE_ESCAPE_CARRY_INDEX("create_escape_carry_index", "file: inout pointer char, n: sint64, escape_carry_index: inout pointer char", "com/koenv/gpjson/kernels/create_escape_carry_index.cu"),
    CREATE_ESCAPE_INDEX("create_escape_index", "file: inout pointer char, n: sint64, escape_carry_index: inout pointer char, escape_index: inout pointer sint64, escape_index_size: sint64", "com/koenv/gpjson/kernels/create_escape_index.cu"),
    CREATE_QUOTE_INDEX("create_quote_index", "file: inout pointer char, n: sint64, escape_index: inout pointer sint64, quote_index: inout pointer sint64, quote_carry_index: inout pointer char, quote_index_size: sint64", "com/koenv/gpjson/kernels/create_quote_index.cu"),
    CREATE_STRING_INDEX("create_string_index", "n: sint64, quote_index: inout pointer sint64, quote_counts: inout pointer char", "com/koenv/gpjson/kernels/create_string_index.cu"),
    CREATE_LEVELED_BITMAPS_CARRY_INDEX("create_leveled_bitmaps_carry_index", "file: inout pointer char, n: sint64, string_index: inout pointer sint64, level_carry_index: inout pointer sint8", "com/koenv/gpjson/kernels/create_leveled_bitmaps_carry_index.cu"),
    CREATE_LEVELED_BITMAPS("create_leveled_bitmaps", "file: inout pointer char, n: sint64, string_index: inout pointer sint64, carry_index: inout pointer sint8, leveled_bitmaps_index: inout pointer sint64, leveled_bitmaps_index_size: sint64, level_size: sint64, num_levels: sint32", "com/koenv/gpjson/kernels/create_leveled_bitmaps.cu"),
    FIND_VALUE("find_value", "file: inout pointer char, n: sint64, new_line_index: in pointer sint64, new_line_index_size: sint64, string_index: in pointer sint64, leveled_bitmaps_index: in pointer sint64, leveled_bitmaps_index_size: sint64, level_size: sint64, num_levels: sint32, query: in pointer uint8, query_size: sint32, result: out pointer sint64", "com/koenv/gpjson/kernels/find_value.cu"),
    CREATE_COMBINED_ESCAPE_CARRY_NEWLINE_COUNT_INDEX("create_combined_escape_carry_newline_count_index", "file: inout pointer char, n: sint64, escape_carry_index: inout pointer char, newline_count_index: inout pointer sint32", "com/koenv/gpjson/kernels/create_combined_escape_carry_newline_count_index.cu"),
    CREATE_COMBINED_ESCAPE_NEWLINE_INDEX("create_combined_escape_newline_index", "file: inout pointer char, n: sint64, escape_carry_index: inout pointer char, newline_count_index: inout pointer sint32, escape_index: inout pointer sint64, escape_index_size: sint64, newline_index: inout pointer sint64", "com/koenv/gpjson/kernels/create_combined_escape_newline_index.cu");

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
