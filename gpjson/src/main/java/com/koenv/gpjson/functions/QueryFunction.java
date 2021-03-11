package com.koenv.gpjson.functions;

import com.jayway.jsonpath.JsonPath;
import com.jayway.jsonpath.ReadContext;
import com.jayway.jsonpath.internal.path.CompiledPath;
import com.jayway.jsonpath.internal.path.PathCompiler;
import com.jayway.jsonpath.internal.path.PathToken;
import com.jayway.jsonpath.internal.path.RootPathToken;
import com.koenv.gpjson.GPJSONException;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.exception.AbstractTruffleException;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import net.minidev.json.JSONAwareEx;
import net.minidev.json.JSONStyle;

import java.io.File;
import java.lang.reflect.Field;

public class QueryFunction extends Function {
    public QueryFunction() {
        super("query");
    }

    @Override
    @CompilerDirectives.TruffleBoundary
    public Object call(Object[] arguments) throws UnsupportedTypeException, ArityException {
        checkArgumentLength(arguments, 2);
        String filename = expectString(arguments[0], "expected filename");
        String query = expectString(arguments[1], "expected query");

        ReadContext ctx;
        try {
            ctx = JsonPath.parse(new File(filename));
        } catch (Exception e) {
            throw new GPJSONException("Failed to parse JSON", e, AbstractTruffleException.UNLIMITED_STACK_TRACE, null);
        }

        JsonPath path;
        try {
            path = JsonPath.compile(query);
        } catch (Exception e) {
            throw new GPJSONException("Failed to parse JSON path", e, AbstractTruffleException.UNLIMITED_STACK_TRACE, null);
        }

        CompiledPath jsonPath = (CompiledPath) PathCompiler.compile(query);

        System.out.println(jsonPath.isDefinite());

        try {
            Field rootField = CompiledPath.class.getDeclaredField("root");
            rootField.setAccessible(true);

            RootPathToken token = (RootPathToken) rootField.get(jsonPath);

            Field nextField = PathToken.class.getDeclaredField("next");
            nextField.setAccessible(true);

            PathToken next = token;
            while (next != null) {
                System.out.printf("%s: %s%n", next, next.getClass().getName());

                next = (PathToken) nextField.get(next);
            }
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }

        Object result;
        try {
            result = ctx.read(path);
        } catch (Exception e) {
            throw new GPJSONException("Failed to read JSON", e, AbstractTruffleException.UNLIMITED_STACK_TRACE, null);
        }

        if (result instanceof JSONAwareEx) {
            return ((JSONAwareEx) result).toJSONString(JSONStyle.NO_COMPRESS);
        }

        return result;
    }
}
