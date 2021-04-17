package com.koenv.gpjson.jsonpath;

public class JSONPathLexer {
    private final String string;
    private int position = -1;

    public JSONPathLexer(String string) {
        this.string = string;
    }

    public char next() {
        if (!hasNext()) {
            throw new JSONPathException("Expected character, got EOF at " + position);
        }

        return this.string.charAt(++position);
    }

    public char peek() {
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

    public void expectChar(char c) {
        char nextChar = next();
        if (c != nextChar) {
            throw new JSONPathException("Expected character '" + c + "', got character '" + nextChar + "' at " + position);
        }
    }
}
