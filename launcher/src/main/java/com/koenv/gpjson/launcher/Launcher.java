package com.koenv.gpjson.launcher;

import org.graalvm.polyglot.*;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

public class Launcher {
    private static final String GPJSON = "gpjson";

    public static void main(String[] args) throws IOException {
        Source source;
        Map<String, String> options = new HashMap<>();
        String file = null;
        for (String arg : args) {
            if (!parseOption(options, arg)) {
                if (file == null) {
                    file = arg;
                }
            }
        }

        if (file == null) {
            source = Source.newBuilder(GPJSON, new InputStreamReader(System.in), "<stdin>").build();
        } else {
            source = Source.newBuilder(GPJSON, new File(file)).build();
        }

        System.exit(executeSource(source, System.in, System.out, options));
    }

    private static int executeSource(Source source, InputStream in, PrintStream out, Map<String, String> options) {
        Context context;
        PrintStream err = System.err;
        try {
            context = Context.newBuilder(GPJSON)
                    .in(in).out(out)
                    .allowPolyglotAccess(PolyglotAccess.ALL)
                    .allowNativeAccess(true)
                    .options(options).build();
        } catch (IllegalArgumentException e) {
            err.println(e.getMessage());
            return 1;
        }
        out.println("== running on " + context.getEngine());

        try {
            Value result = context.eval(source);
            if (!result.isNull()) {
                out.println(result.toString());
            }
            return 0;
        } catch (PolyglotException ex) {
            if (ex.isInternalError()) {
                // for internal errors we print the full stack trace
                ex.printStackTrace();
            } else {
                err.println(ex.getMessage());
            }
            return 1;
        } finally {
            context.close();
        }
    }

    private static boolean parseOption(Map<String, String> options, String arg) {
        if (arg.length() <= 2 || !arg.startsWith("--")) {
            return false;
        }
        int eqIdx = arg.indexOf('=');
        String key;
        String value;
        if (eqIdx < 0) {
            key = arg.substring(2);
            value = null;
        } else {
            key = arg.substring(2, eqIdx);
            value = arg.substring(eqIdx + 1);
        }

        if (value == null) {
            value = "true";
        }
        int index = key.indexOf('.');
        String group = key;
        if (index >= 0) {
            group = group.substring(0, index);
        }
        options.put(key, value);
        return true;
    }
}
