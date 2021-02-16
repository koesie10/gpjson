package com.koenv.gpjson;

import com.oracle.truffle.api.exception.AbstractTruffleException;
import com.oracle.truffle.api.nodes.Node;

import java.util.Arrays;
import java.util.Optional;

public class GPJSONException extends AbstractTruffleException {
    public GPJSONException() {
    }

    public GPJSONException(Node location) {
        super(location);
    }

    public GPJSONException(String message) {
        super(message);
    }

    public GPJSONException(String message, Node location) {
        super(message, location);
    }

    public GPJSONException(String message, Throwable cause, int stackTraceElementLimit, Node location) {
        super(message, cause, stackTraceElementLimit, location);
    }

    public GPJSONException(Throwable cause) {
        super(null, cause, AbstractTruffleException.UNLIMITED_STACK_TRACE, null);
    }

    public GPJSONException(int errorCode, String[] functionName) {
        this("CUDA error " + errorCode + " in " + format(functionName));
    }

    public GPJSONException(int errorCode, String message, String[] functionName) {
        this(message + '(' + errorCode + ") in " + format(functionName));
    }

    public static String format(String... name) {
        Optional<String> result = Arrays.stream(name).reduce((a, b) -> a + "::" + b);
        return result.orElse("<empty>");
    }
}
