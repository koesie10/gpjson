package com.koenv.gpjson.jsonpath;

public class UnsupportedJSONPathException extends JSONPathException {
    public UnsupportedJSONPathException(String message) {
        super(message);
    }

    public UnsupportedJSONPathException(String message, Throwable cause) {
        super(message, cause);
    }
}
