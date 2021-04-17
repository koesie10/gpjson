package com.koenv.gpjson.function;

import com.koenv.gpjson.jsonpath.JSONPathLexer;
import com.koenv.gpjson.jsonpath.JSONPathParser;
import com.koenv.gpjson.jsonpath.JSONPathResult;
import com.koenv.gpjson.sequential.Sequential;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.PolyglotAccess;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;

@Disabled("Currently only used for manual testing")
public class QueryGPUFunctionTest {
    private static Context context = Context.newBuilder("nfi", "gpjson")
            .allowPolyglotAccess(PolyglotAccess.ALL)
            .allowNativeAccess(true)
            .build();

    @BeforeEach
    void setUp() {
        context.enter();
    }

    @AfterEach
    public void tearDown() {
        context.leave();
    }

    @Test
    public void twitterSmall() {
        System.out.println(context.eval("gpjson", "jsonpath").invokeMember("queryGPU", "out/twitter_small_records_smaller.json", "$.user.lang"));
    }

    @Test
    public void twitterSmallSequential() throws IOException {
        byte[] file = Files.readAllBytes(Paths.get("out/twitter_really_small.ldjson"));
        JSONPathResult compiledQuery = new JSONPathParser(new JSONPathLexer("$.user.lang")).compile();

        long[] newlineIndex = Sequential.createNewlineIndex(file);
        long[] stringIndex = Sequential.createStringIndex(file);
        long[] leveledBitmapsIndex = Sequential.createLeveledBitmapsIndex(file, stringIndex, compiledQuery.getMaxDepth());
        long[] result = Sequential.findValue(file, newlineIndex, stringIndex, leveledBitmapsIndex, compiledQuery);

        StringBuilder returnValue = new StringBuilder();

        for (long value : result) {
            returnValue.append(value);

            if (value > -1) {
                returnValue.append(": ");

                for (int m = 0; m < 8; m++) {
                    returnValue.append((char)file[(int) value + m]);
                }
            }

            returnValue.append('\n');
        }

        Files.write(Paths.get("result.json"), returnValue.toString().getBytes(StandardCharsets.UTF_8));
    }
}
