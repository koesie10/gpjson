package com.koenv.gpjson.jsonpath;

import java.io.StringWriter;
import java.util.function.Predicate;

public class JSONPathParser {
    private final JSONPathScanner scanner;

    private final IRByteBuffer output = new IRByteBuffer();
    private int maxLevel = 0;

    public JSONPathParser(JSONPathScanner scanner) {
        this.scanner = scanner;
    }

    public JSONPathResult compile() {
        scanner.expectChar('$');

        compileNextExpression();

        // Denote end
        output.writeByte(0x00);

        return new JSONPathResult(output, maxLevel);
    }

    private void compileNextExpression() {
        char c = scanner.peek();
        switch (c) {
            case '.':
                compileDotExpression();
                break;
            case '[':
                compileIndexExpression();
                break;
            default:
                throw new JSONPathException("Unsupported expression type '" + c + "'");
        }
    }

    private void compileDotExpression() {
        scanner.expectChar('.');

        if (scanner.peek() == '.') {
            throw new JSONPathException("Unsupported recursive descent path");
        }

        String property = readProperty();
        if (property.isEmpty()) {
            throw new JSONPathException("Unexpected empty property");
        }

        createPropertyIR(property);

        if (scanner.hasNext()) {
            compileNextExpression();
        }
    }

    private void compileIndexExpression() {
        scanner.expectChar('[');

        if (scanner.peek() == '\'' || scanner.peek() == '"') {
            String property = readQuotedString();
            if (property.isEmpty()) {
                throw new JSONPathException("Unexpected empty property");
            }

            createPropertyIR(property);
        } else if (scanner.peek() >= '0' && scanner.peek() <= '9') {
            int index = readInteger(c -> c == ']');

            createIndexIR(index);
        } else {
            throw scanner.errorNext("Unexpected character in index, expected ', \", or an integer");
        }

        scanner.expectChar(']');

        if (scanner.hasNext()) {
            compileNextExpression();
        }
    }

    private void createPropertyIR(String propertyName) {
        // Type = property
        output.writeByte(0x01);
        output.writeString(propertyName);

        maxLevel++;
    }

    private void createIndexIR(int index) {
        // Type = index
        output.writeByte(0x02);
        output.writeVarInt(index);

        maxLevel++;
    }

    private String readProperty() {
        int startPosition = scanner.position();

        while (scanner.hasNext()) {
            char c = scanner.peek();
            if (c == ' ') {
                throw new JSONPathException("Unexpected space");
            } else if (c == '.' || c == '[') {
                break;
            }

            scanner.next();
        }

        int endPosition = scanner.position();

        return scanner.substring(startPosition, endPosition);
    }

    private String readQuotedString() {
        char quoteCharacter = scanner.next();
        if (quoteCharacter != '\'' && quoteCharacter != '"') {
            throw new JSONPathException("Invalid quoted string");
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

        String escapedString = scanner.substring(startPosition, endPosition);
        return unescape(escapedString);
    }

    private int readInteger(Predicate<Character> isEndCharacter) {
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

    /**
     * Licensed under the Apache 2.0 license
     *
     * @link https://github.com/json-path/JsonPath/blob/0dca7ea7f8fcd0302d326b9206dc119c10f76c52/json-path/src/main/java/com/jayway/jsonpath/internal/Utils.java#L172
     */
    private static String unescape(String str) {
        if (str == null) {
            return null;
        }
        int len = str.length();
        StringWriter writer = new StringWriter(len);
        StringBuilder unicode = new StringBuilder(4);
        boolean hadSlash = false;
        boolean inUnicode = false;
        for (int i = 0; i < len; i++) {
            char ch = str.charAt(i);
            if (inUnicode) {
                unicode.append(ch);
                if (unicode.length() == 4) {
                    try {
                        int value = Integer.parseInt(unicode.toString(), 16);
                        writer.write((char) value);
                        unicode.setLength(0);
                        inUnicode = false;
                        hadSlash = false;
                    } catch (NumberFormatException nfe) {
                        throw new JSONPathException("Unable to parse unicode value: " + unicode, nfe);
                    }
                }
                continue;
            }
            if (hadSlash) {
                hadSlash = false;
                switch (ch) {
                    case '\\':
                        writer.write('\\');
                        break;
                    case '\'':
                        writer.write('\'');
                        break;
                    case '\"':
                        writer.write('"');
                        break;
                    case 'r':
                        writer.write('\r');
                        break;
                    case 'f':
                        writer.write('\f');
                        break;
                    case 't':
                        writer.write('\t');
                        break;
                    case 'n':
                        writer.write('\n');
                        break;
                    case 'b':
                        writer.write('\b');
                        break;
                    case 'u':
                    {
                        inUnicode = true;
                        break;
                    }
                    default :
                        writer.write(ch);
                        break;
                }
                continue;
            } else if (ch == '\\') {
                hadSlash = true;
                continue;
            }
            writer.write(ch);
        }
        if (hadSlash) {
            writer.write('\\');
        }
        return writer.toString();
    }
}
