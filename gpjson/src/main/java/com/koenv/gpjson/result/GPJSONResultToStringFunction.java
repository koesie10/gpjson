package com.koenv.gpjson.result;

import com.koenv.gpjson.GPJSONException;
import com.koenv.gpjson.functions.Function;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.exception.AbstractTruffleException;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

public class GPJSONResultToStringFunction extends Function {
    private final GPJSONResultValue value;

    protected GPJSONResultToStringFunction(GPJSONResultValue value) {
        super("toString");
        this.value = value;
    }

    @Override
    @CompilerDirectives.TruffleBoundary
    public Object call(Object[] arguments) throws UnsupportedTypeException, ArityException {
        checkArgumentLength(arguments, 0);

        StringBuilder returnValue = new StringBuilder();

        for (long i = 0; i < value.getArraySize(); i++) {
            Object element;
            try {
                element = value.readArrayElement(i);
            } catch (UnsupportedMessageException | InvalidArrayIndexException e) {
                throw new GPJSONException("Failed to read element", e, AbstractTruffleException.UNLIMITED_STACK_TRACE, null);
            }

            returnValue.append(element);

            returnValue.append('\n');
        }

        return returnValue.toString();
    }
}
