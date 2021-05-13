package com.koenv.gpjson.jsonpath;

import java.util.Stack;

public class JSONPathScanner {
    private final String string;
    private int position = -1;

    private final Stack<Integer> markedPositions = new Stack<>();

    public JSONPathScanner(String string) {
        this.string = string;
    }

    public char next() throws JSONPathException {
        if (!hasNext()) {
            throw new JSONPathException("Expected character, got EOF at " + position);
        }

        return this.string.charAt(++position);
    }

    public char peek() throws JSONPathException {
        if (!hasNext()) {
            throw new JSONPathException("Expected character, got EOF at " + position);
        }

        return this.string.charAt(position + 1);
    }

    public boolean hasNext() {
        return position + 1 < string.length();
    }

    public int position() {
        return position;
    }

    public String substring(int start, int end) {
        return string.substring(start + 1, end + 1);
    }

    public void expectChar(char c) throws JSONPathException {
        char nextChar = next();
        if (c != nextChar) {
            throw new JSONPathException("Expected character '" + c + "', got character '" + nextChar + "' at " + position);
        }
    }

    public JSONPathException error(String errorMessage) {
        char currentChar = this.string.charAt(position);

        return new JSONPathException(errorMessage + " at " + position + " ('" + currentChar + "')");
    }

    public JSONPathException errorNext(String errorMessage) {
        char currentChar = this.string.charAt(position + 1);

        return new JSONPathException(errorMessage + " at " + (position + 1) + " ('" + currentChar + "')");
    }

    public JSONPathException unsupported(String errorMessage) {
        char currentChar = this.string.charAt(position);

        return new UnsupportedJSONPathException(errorMessage + " at " + (position + 1) + " ('" + currentChar + "')");
    }

    public JSONPathException unsupportedNext(String errorMessage) {
        char currentChar = this.string.charAt(position + 1);

        return new UnsupportedJSONPathException(errorMessage + " at " + (position + 1) + " ('" + currentChar + "')");
    }

    public void mark() {
        markedPositions.push(this.position);
    }

    public void reset() {
        this.position = markedPositions.pop();
    }
}
