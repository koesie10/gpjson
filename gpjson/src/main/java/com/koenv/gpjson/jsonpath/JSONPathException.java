package com.koenv.gpjson.jsonpath;

public class JSONPathException extends RuntimeException {
    public JSONPathException() {
    }

    public JSONPathException(String message) {
        super(message);
    }

    public JSONPathException(Throwable cause) {
        super(cause);
    }
}
