package com.koenv.gpjson.jsonpath;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class JSONPathParserTest {
    @Test
    public void testSimple() {
        JSONPathLexer lexer = new JSONPathLexer("$.user.lang");
        JSONPathParser parser = new JSONPathParser(lexer);

        IRByteBuffer buffer = parser.compile();

        assertArrayEquals(new byte[]{0x01, 0x04, 'u', 's', 'e', 'r', 0x01, 0x04, 'l', 'a', 'n', 'g', 0x00}, buffer.toByteArray());
    }
}
