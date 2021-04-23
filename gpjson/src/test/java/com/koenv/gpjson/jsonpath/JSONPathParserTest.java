package com.koenv.gpjson.jsonpath;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class JSONPathParserTest {
    @ParameterizedTest(name = "[{index}] query = \"{0}\"")
    @MethodSource("simpleProvider")
    public void testSimple(String query, byte[] ir, int maxDepth) {
        JSONPathScanner scanner = new JSONPathScanner(query);
        JSONPathParser parser = new JSONPathParser(scanner);

        JSONPathResult result = parser.compile();

        assertArrayEquals(ir, result.getIr().toByteArray());
        assertEquals(maxDepth, result.getMaxDepth());
    }

    private static Stream<Arguments> simpleProvider() {
        return Stream.of(
                Arguments.of("$.user.lang", new byte[]{0x01, 0x04, 'u', 's', 'e', 'r', 0x01, 0x04, 'l', 'a', 'n', 'g', 0x00}, 2),
                Arguments.of("$['user'][\"lang\"]", new byte[]{0x01, 0x04, 'u', 's', 'e', 'r', 0x01, 0x04, 'l', 'a', 'n', 'g', 0x00}, 2),
                Arguments.of("$[1]", new byte[]{0x02, 0x01, 0x00}, 1),
                Arguments.of("$['us\\'er'][\"lang\"]", new byte[]{0x01, 0x05, 'u', 's', '\'', 'e', 'r', 0x01, 0x04, 'l', 'a', 'n', 'g', 0x00}, 2)
        );
    }
}
