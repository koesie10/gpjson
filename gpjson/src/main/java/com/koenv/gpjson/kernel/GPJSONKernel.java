package com.koenv.gpjson.kernel;

public enum GPJSONKernel {
    DISCOVER_STRUCTURE("discover_structure", "pointer, sint64", "com/koenv/gpjson/kernels/discover_structure.cu");

    private final String name;
    private final String nidlSignature;
    private final String filename;

    GPJSONKernel(String name, String nidlSignature, String filename) {
        this.name = name;
        this.nidlSignature = nidlSignature;
        this.filename = filename;
    }

    public String getName() {
        return name;
    }

    public String getNidlSignature() {
        return nidlSignature;
    }

    public String getFilename() {
        return filename;
    }
}
