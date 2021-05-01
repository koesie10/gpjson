package com.koenv.gpjson.jsonpath;

public class JSONPathResult {
    private final IRByteOutputBuffer ir;
    private final int maxDepth;

    public JSONPathResult(IRByteOutputBuffer ir, int maxDepth) {
        this.ir = ir;
        this.maxDepth = maxDepth;
    }

    public IRByteOutputBuffer getIr() {
        return ir;
    }

    public int getMaxDepth() {
        return maxDepth;
    }
}
