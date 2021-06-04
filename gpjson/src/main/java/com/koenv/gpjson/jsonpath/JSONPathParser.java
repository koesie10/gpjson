package com.koenv.gpjson.jsonpath;

import java.util.function.Predicate;

public class JSONPathParser {
    private final JSONPathScanner scanner;

    private final IRByteOutputBuffer output = new IRByteOutputBuffer();
    private final IRBuilder ir = new IRBuilder(output);
    private int maxLevel = 0;

    public JSONPathParser(JSONPathScanner scanner) {
        this.scanner = scanner;
    }

    public JSONPathParser(String string) {
        this(new JSONPathScanner(string));
    }

    public JSONPathResult compile() throws JSONPathException {
        scanner.expectChar('$');

        compileNextExpression();

        ir.storeResult();
        ir.end();

        return new JSONPathResult(output, maxLevel, ir.getNumResultStores());
    }

    private void compileNextExpression() throws JSONPathException {
        char c = scanner.peek();
        switch (c) {
            case '.':
                compileDotExpression();
                break;
            case '[':
                compileIndexExpression();
                break;
            default:
                throw scanner.unsupportedNext("Unsupported expression type");
        }
    }

    private void compileDotExpression() throws JSONPathException {
        scanner.expectChar('.');

        if (scanner.peek() == '.') {
            throw scanner.unsupportedNext("Unsupported recursive descent");
        }

        String property = readProperty();
        if (property.isEmpty()) {
            throw scanner.error("Unexpected empty property");
        }

        createPropertyIR(property);

        if (scanner.hasNext()) {
            compileNextExpression();
        }
    }

    private void compileIndexExpression() throws JSONPathException {
        scanner.expectChar('[');

        if (scanner.peek() == '\'' || scanner.peek() == '"') {
            String property = readQuotedString();
            if (property.isEmpty()) {
                throw scanner.error("Unexpected empty property");
            }

            createPropertyIR(property);
        } else if (scanner.peek() >= '0' && scanner.peek() <= '9') {
            int index = readInteger(c -> c == ']' || c == ':' || c == ',');

            switch (scanner.peek()) {
                case ':':
                    compileIndexRangeExpression(index);

                    // The index range expression will parse the rest
                    return;
                case ',':
                    throw scanner.unsupportedNext("Unsupported multiple index expression");
                case ']':
                    createIndexIR(index);
                    break;
            }
        } else if (scanner.peek() == '*') {
            throw scanner.unsupportedNext("Unsupported wildcard expression");
        } else if (scanner.peek() == '?') {
            compileFilterExpression();
        } else {
            throw scanner.errorNext("Unexpected character in index, expected ', \", or an integer");
        }

        scanner.expectChar(']');

        if (scanner.hasNext()) {
            compileNextExpression();
        }
    }

    private void compileIndexRangeExpression(int startIndex) throws JSONPathException {
        scanner.expectChar(':');

        int endIndex = readInteger(c -> c == ']');

        scanner.expectChar(']');

        int maxMaxLevel = maxLevel;

        for (int i = startIndex; i < endIndex; i++) {
            int startLevel = ir.getCurrentLevel();

            ir.index(i);
            ir.down();

            scanner.mark();

            int currentMaxLevel = maxLevel;

            if (scanner.hasNext()) {
                compileNextExpression();
            }

            maxMaxLevel = Math.max(maxLevel, maxMaxLevel);

            maxLevel = currentMaxLevel;

            if (i == endIndex - 1) {
                break;
            } else {
                ir.storeResult();
            }

            scanner.reset();

            int endLevel = ir.getCurrentLevel();

            for (int j = 0; j < endLevel - startLevel; j++) {
                ir.up();
            }
        }

        maxLevel = maxMaxLevel + 1;
    }

    private void compileFilterExpression() throws JSONPathException {
        scanner.expectChar('?');
        scanner.expectChar('(');
        scanner.expectChar('@');

        while (scanner.skipIfChar(' ')) {
            // Skip whitespace
        }

        switch (scanner.peek()) {
            case '=':
                scanner.expectChar('=');
                scanner.expectChar('=');

                while (scanner.skipIfChar(' ')) {
                    // Skip whitespace
                }

                String equalTo = readQuotedString();

                ir.expressionStringEquals(equalTo);

                break;
            default:
                throw scanner.unsupportedNext("Unsupported character for expression");
        }

        scanner.expectChar(')');
    }

    private void createPropertyIR(String propertyName) {
        ir.property(propertyName);
        ir.down();

        maxLevel++;
    }

    private void createIndexIR(int index) {
        ir.index(index);
        ir.down();

        maxLevel++;
    }

    private String readProperty() throws JSONPathException {
        int startPosition = scanner.position();

        while (scanner.hasNext()) {
            char c = scanner.peek();
            if (c == ' ') {
                throw scanner.errorNext("Unexpected space");
            } else if (c == '.' || c == '[') {
                break;
            }

            scanner.next();
        }

        int endPosition = scanner.position();

        return scanner.substring(startPosition, endPosition);
    }

    private String readQuotedString() throws JSONPathException {
        char quoteCharacter = scanner.next();
        if (quoteCharacter != '\'' && quoteCharacter != '"') {
            throw scanner.error("Invalid quoted string");
        }

        int startPosition = scanner.position();

        boolean escaped = false;

        while (scanner.hasNext()) {
            char c = scanner.peek();
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == quoteCharacter) {
                break;
            }

            scanner.next();
        }

        int endPosition = scanner.position();

        scanner.expectChar(quoteCharacter);

        return scanner.substring(startPosition, endPosition);
    }

    private int readInteger(Predicate<Character> isEndCharacter) throws JSONPathException {
        int startPosition = scanner.position();

        while (scanner.hasNext()) {
            char c = scanner.peek();
            if (c >= '0' && c <= '9') {
                scanner.next();
                continue;
            } else if (isEndCharacter.test(c)) {
                break;
            }

            throw scanner.error("Invalid integer");
        }

        int endPosition = scanner.position();

        String str = scanner.substring(startPosition, endPosition);
        return Integer.parseInt(str);
    }
}
