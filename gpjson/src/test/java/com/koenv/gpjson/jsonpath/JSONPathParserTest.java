package com.koenv.gpjson.jsonpath;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class JSONPathParserTest {
    @Test
    public void testSimple() {
        JSONPathLexer lexer = new JSONPathLexer("$.user.lang");
        JSONPathParser parser = new JSONPathParser(lexer);

        JSONPathResult result = parser.compile();

        assertArrayEquals(new byte[]{0x01, 0x04, 'u', 's', 'e', 'r', 0x01, 0x04, 'l', 'a', 'n', 'g', 0x00}, result.getIr().toByteArray());
        assertEquals(2, result.getMaxDepth());
    }
}
