package com.koenv.gpjson.functions;

import com.jayway.jsonpath.JsonPath;
import com.jayway.jsonpath.ReadContext;
import com.koenv.gpjson.GPJSONException;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.exception.AbstractTruffleException;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import net.minidev.json.JSONAwareEx;
import net.minidev.json.JSONStyle;

import java.io.File;

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
