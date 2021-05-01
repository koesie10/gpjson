package com.koenv.gpjson.function;

import com.koenv.gpjson.GPJSONTest;
import com.koenv.gpjson.jsonpath.JSONPathException;
import com.koenv.gpjson.jsonpath.JSONPathParser;
import com.koenv.gpjson.jsonpath.JSONPathResult;
import com.koenv.gpjson.jsonpath.JSONPathScanner;
import com.koenv.gpjson.sequential.Sequential;
import org.graalvm.polyglot.Value;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class QueryFunctionTest extends GPJSONTest {
    @BeforeEach
    void setUp() {
        context.enter();
    }

    @AfterEach
    public void tearDown() {
        context.leave();
    }

    @Test
    @Disabled("Currently only used for manual testing")
    public void twitterSmall() {
        System.out.println(context.eval("gpjson", "jsonpath").invokeMember("query", "out/twitter_really_small.ldjson", "$.user.lang"));
    }

    @Test
    @Disabled("Currently only used for manual testing")
    public void twitterSmallSequential() throws IOException, JSONPathException {
        byte[] file = Files.readAllBytes(Paths.get("out/twitter_really_small.ldjson"));
        JSONPathResult compiledQuery = new JSONPathParser(new JSONPathScanner("$.user.lang")).compile();

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

    @Test
    public void keyNotFoundInCurrentLevel() throws IOException {
        Path tempFile = createTemporaryFile("query_gpu_function/not_found_in_current_level.ldjson");

        try {
            Value result = context.eval("gpjson", "jsonpath").invokeMember("query", tempFile.toAbsolutePath().toString(), "$.a.a");

            long valueCount = 0;

            for (int i = 0; i < result.getArraySize(); i++) {
                Value path = result.getArrayElement(i);

                for (int j = 0; j < path.getArraySize(); j++) {
                    Value line = path.getArrayElement(j);
                    valueCount += line.getArraySize();
                }
            }

            assertEquals(0, valueCount);
        } finally {
            Files.deleteIfExists(tempFile);
        }
    }

    private Path createTemporaryFile(String name) throws IOException {
        Path tempFile = Files.createTempFile("", ".ldjson");

        try (InputStream is = getClass().getClassLoader().getResourceAsStream(name)) {
            if (is == null) {
                throw new FileNotFoundException("File " + name + " not found in resources");
            }

            ByteArrayOutputStream result = new ByteArrayOutputStream();
            byte[] buffer = new byte[1 << 20];
            for (int length; (length = is.read(buffer)) != -1; ) {
                result.write(buffer, 0, length);
            }

            Files.write(tempFile, result.toByteArray());
        }

        return tempFile;
    }
}
