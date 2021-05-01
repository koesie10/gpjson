package com.koenv.gpjson.jsonpath;

public class JSONPathException extends Exception {
    public JSONPathException(String message) {
        super(message);
    }

    public JSONPathException(String message, Throwable cause) {
        super(message, cause);
    }
}
