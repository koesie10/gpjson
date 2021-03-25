package com.koenv.gpjson.gpu;

public enum CUDAMemcpyKind {
    HOST_TO_HOST(0),
    HOST_TO_DEVICE(1),
    DEVICE_TO_HOST(2),
    DEVICE_TO_DEVICE(3),
    DEFAULT(4);

    private final int kind;

    CUDAMemcpyKind(int kind) {
        this.kind = kind;
    }

    public int getKind() {
        return kind;
    }
}
