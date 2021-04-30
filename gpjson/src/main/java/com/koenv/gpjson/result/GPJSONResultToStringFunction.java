package com.koenv.gpjson.result;

import com.koenv.gpjson.GPJSONException;
import com.koenv.gpjson.debug.GPUUtils;
import com.koenv.gpjson.functions.Function;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.exception.AbstractTruffleException;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

import java.io.IOException;
import java.nio.file.Files;

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

        byte[] file;
        try {
            file = Files.readAllBytes(value.getFilePath());
        } catch (IOException e) {
            throw new GPJSONException("Failed to read file", e, AbstractTruffleException.UNLIMITED_STACK_TRACE, null);
        }

        long[] returnValues = value.getValue();

        for (int i = 0; i < value.getNumberOfElements(); i++) {
            long valueStart = returnValues[i * 2];
            long valueEnd = returnValues[i * 2 + 1];

            returnValue.append(valueStart);

            if (valueStart > -1) {
                returnValue.append(": ");

                for (int m = (int) valueStart; m < valueEnd; m++) {
                    returnValue.append((char) file[m]);
                }
            }

            returnValue.append('\n');
        }

        return returnValue.toString();
    }
}
