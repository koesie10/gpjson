package com.koenv.gpjson.jsonpath;

public class JSONPathParser {
    private final JSONPathLexer lexer;

    private final IRByteBuffer output = new IRByteBuffer();
    private int maxLevel = 0;

    public JSONPathParser(JSONPathLexer lexer) {
        this.lexer = lexer;
    }

    public JSONPathResult compile() {
        lexer.expectChar('$');

        compileNextExpression();

        // Denote end
        output.writeByte(0x00);

        return new JSONPathResult(output, maxLevel);
    }

    private void compileNextExpression() {
        char c = lexer.peek();
        switch (c) {
            case '.':
                compileDotExpression();
                break;
            default:
                throw new JSONPathException("Unsupported expression type '" + c + "'");
        }
    }

    private void compileDotExpression() {
        lexer.expectChar('.');

        if (lexer.peek() == '.') {
            throw new JSONPathException("Unsupported recursive descent path");
        }

        String property = readProperty();
        if (property.isEmpty()) {
            throw new JSONPathException("Unexpected empty property");
        }

        // Type = dot
        output.writeByte(0x01);
        output.writeString(property);

        maxLevel++;

        if (lexer.hasNext()) {
            compileNextExpression();
        }
    }

    private String readProperty() {
        int startPosition = lexer.position();

        while (lexer.hasNext()) {
            char c = lexer.peek();
            if (c == ' ') {
                throw new JSONPathException("Unexpected space");
            } else if (c == '.' || c == '[') {
                break;
            }

            lexer.next();
        }

        int endPosition = lexer.position();

        return lexer.substring(startPosition, endPosition);
    }
}
