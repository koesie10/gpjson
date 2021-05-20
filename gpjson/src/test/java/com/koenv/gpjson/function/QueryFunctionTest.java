package com.koenv.gpjson.function;

import com.koenv.gpjson.GPJSONTest;
import com.koenv.gpjson.jsonpath.*;
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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

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
        Value result = context.eval("gpjson", "jsonpath").invokeMember("query", "out/twitter_very_small.ldjson", "$.user.lang");
        List<Value> values = resultToValues(result);

        assertEquals(1, values.size());

        System.out.println(values.get(0));
    }

    @Test
    @Disabled("Currently only used for manual testing")
    public void twitterSmallSequential() throws IOException, JSONPathException {
        byte[] file = Files.readAllBytes(Paths.get("out/twitter_very_small.ldjson"));
        JSONPathResult compiledQuery = new JSONPathParser(new JSONPathScanner("$.user.values[2]")).compile();

        IRVisitor.accept(compiledQuery.getIr().toReadable(), new PrintingIRVisitor(System.out));

        long[] newlineIndex = Sequential.createNewlineIndex(file);
        long[] stringIndex = Sequential.createStringIndex(file);
        long[] leveledBitmapsIndex = Sequential.createLeveledBitmapsIndex(file, stringIndex, compiledQuery.getMaxDepth());
        long[] result = Sequential.findValue(file, newlineIndex, stringIndex, leveledBitmapsIndex, compiledQuery);

        System.out.println(Arrays.toString(result));

        StringBuilder returnValue = new StringBuilder();

        for (int i = 0; i < result.length; i += 2) {
            long start = result[i];
            long end = result[i + 1];

            returnValue.append(start);

            if (start > -1) {
                returnValue.append(": ");

                for (long m = start; m < end; m++) {
                    returnValue.append((char) file[(int) m]);
                }
            }

            returnValue.append('\n');
        }

        System.out.println(returnValue);
    }

    @Test
    public void simpleNestedStringProperty() throws IOException {
        List<Value> values = simpleQuery("query_gpu_function/simple_single_line.ldjson", "$.user.lang");

        assertEquals(1, values.size());
        assertEquals("\"nl\"", values.get(0).asString());
    }

    @Test
    public void simpleNestedNumberProperty() throws IOException {
        List<Value> values = simpleQuery("query_gpu_function/simple_single_line.ldjson", "$.user.id");

        assertEquals(1, values.size());
        assertEquals("8723", values.get(0).asString());
    }

    @Test
    public void simpleNonExistentLeafProperty() throws IOException {
        List<Value> values = simpleQuery("query_gpu_function/simple_single_line.ldjson", "$.user.foo");

        assertEquals(1, values.size());
        assertTrue(values.get(0).isNull());
    }

    @Test
    public void simpleNonExistentIntermediateProperty() throws IOException {
        List<Value> values = simpleQuery("query_gpu_function/simple_single_line.ldjson", "$.apple.lang");

        assertEquals(1, values.size());
        assertTrue(values.get(0).isNull());
    }

    @Test
    public void simpleNestedZeroIndex() throws IOException {
        List<Value> values = simpleQuery("query_gpu_function/simple_single_line.ldjson", "$.user.values[0]");

        assertEquals(1, values.size());
        assertEquals("\"12\"", values.get(0).asString());
    }

    @Test
    public void simpleNestedIndex() throws IOException {
        List<Value> values = simpleQuery("query_gpu_function/simple_single_line.ldjson", "$.user.values[2]");

        assertEquals(1, values.size());
        assertEquals("14", values.get(0).asString());
    }

    @Test
    public void simpleNonExistentIndex() throws IOException {
        List<Value> values = simpleQuery("query_gpu_function/simple_single_line.ldjson", "$.user.values[8]");

        assertEquals(1, values.size());
        assertTrue(values.get(0).isNull());
    }

    @Test
    public void simpleZeroBasedIndexRange() throws IOException {
        List<Value> values = simpleQuery("query_gpu_function/simple_single_line.ldjson", "$.user.values[0:3]");

        assertEquals(3, values.size());
        assertEquals("\"12\"", values.get(0).asString());
        assertEquals("\"13\"", values.get(1).asString());
        assertEquals("14", values.get(2).asString());
    }

    @Test
    public void simpleNonZeroBasedIndexRange() throws IOException {
        List<Value> values = simpleQuery("query_gpu_function/simple_single_line.ldjson", "$.user.values[2:4]");

        assertEquals(2, values.size());
        assertEquals("14", values.get(0).asString());
        assertEquals("\"15\"", values.get(1).asString());
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

                    for (int k = 0; k < line.getArraySize(); k++) {
                        Value item = line.getArrayElement(k);

                        if (!item.isNull()) {
                            valueCount++;
                        }
                    }
                }
            }

            assertEquals(0, valueCount);
        } finally {
            Files.deleteIfExists(tempFile);
        }
    }

    private List<Value> simpleQuery(String file, String query) throws IOException {
        Path tempFile = createTemporaryFile(file);

        try {
            Value result = context.eval("gpjson", "jsonpath").invokeMember("query", tempFile.toAbsolutePath().toString(), query);

            return resultToValues(result);
        } finally {
            Files.deleteIfExists(tempFile);
        }
    }

    private List<Value> resultToValues(Value result)  {
        List<Value> values = new ArrayList<>();

        for (int i = 0; i < result.getArraySize(); i++) {
            Value path = result.getArrayElement(i);

            for (int j = 0; j < path.getArraySize(); j++) {
                Value line = path.getArrayElement(j);

                for (int k = 0; k < line.getArraySize(); k++) {
                    Value item = line.getArrayElement(k);

                    values.add(item);
                }
            }
        }

        return values;
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
