package com.koenv.gpjson;

import com.oracle.truffle.api.exception.AbstractTruffleException;
import com.oracle.truffle.api.nodes.Node;

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
}
