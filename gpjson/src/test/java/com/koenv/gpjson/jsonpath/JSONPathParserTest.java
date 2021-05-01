package com.koenv.gpjson.jsonpath;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

public class JSONPathParserTest {
    @ParameterizedTest(name = "[{index}] query = \"{0}\"")
    @MethodSource("simpleProvider")
    public void testSimple(String query, byte[] ir, int maxDepth) throws JSONPathException {
        JSONPathParser parser = new JSONPathParser(query);

        JSONPathResult result = parser.compile();

        assertArrayEquals(ir, result.getIr().toByteArray());
        assertEquals(maxDepth, result.getMaxDepth());
    }

    private static Stream<Arguments> simpleProvider() {
        return Stream.of(
                Arguments.of("$.user.lang", new IRBuilder().property("user").property("lang").end().toByteArray(), 2),
                Arguments.of("$['user'][\"lang\"]", new IRBuilder().property("user").property("lang").end().toByteArray(), 2),
                Arguments.of("$[1]", new IRBuilder().index(1).end().toByteArray(), 1),
                Arguments.of("$['us\\'er'][\"lang\"]", new IRBuilder().property("us\\'er").property("lang").end().toByteArray(), 2)
        );
    }

    @ParameterizedTest(name = "[{index}] query = \"{0}\"")
    @MethodSource("unsupportedProvider")
    public void testUnsupported(String query) throws JSONPathException {
        assertThrows(UnsupportedJSONPathException.class, () -> {
            new JSONPathParser(query).compile();
        });
    }

    private static Stream<Arguments> unsupportedProvider() {
        return Stream.of(
                Arguments.of("$.categoryPath[1:3].id"),
                Arguments.of("$.routes[*].legs[*].steps[*].distance.text"),
                Arguments.of("$.meta.view.columns[*].name"),
                Arguments.of("$.claims.P150[*].mainsnak.property")
        );
    }
}
