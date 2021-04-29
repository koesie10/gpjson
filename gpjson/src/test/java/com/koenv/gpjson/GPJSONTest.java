package com.koenv.gpjson;

import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.PolyglotAccess;

import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;

public class GPJSONTest {
    protected static Context context = Context.newBuilder("nfi", "gpjson")
            .allowPolyglotAccess(PolyglotAccess.ALL)
            .allowNativeAccess(true)
            .build();

    protected byte[] readFile(String name) throws IOException {
        try (InputStream is = getClass().getClassLoader().getResourceAsStream(name)) {
            if (is == null) {
                throw new FileNotFoundException("File " + name + " not found in resources");
            }

            ByteArrayOutputStream result = new ByteArrayOutputStream();
            byte[] buffer = new byte[1 << 20];
            for (int length; (length = is.read(buffer)) != -1; ) {
                result.write(buffer, 0, length);
            }
            return result.toByteArray();
        }
    }
}
