package com.koenv.gpjson.kernel;

public enum GPJSONKernel {
    DISCOVER_STRUCTURE("discover_structure", "data: inout pointer char, n: sint64", "com/koenv/gpjson/kernels/discover_structure.cu");

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
