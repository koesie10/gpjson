package com.koenv.gpjson.jsonpath;

public class JSONPathResult {
    private final IRByteBuffer ir;
    private final int maxDepth;

    public JSONPathResult(IRByteBuffer ir, int maxDepth) {
        this.ir = ir;
        this.maxDepth = maxDepth;
    }

    public IRByteBuffer getIr() {
        return ir;
    }

    public int getMaxDepth() {
        return maxDepth;
    }
}
