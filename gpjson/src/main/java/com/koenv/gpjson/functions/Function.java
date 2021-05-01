package com.koenv.gpjson.functions;

import com.oracle.truffle.api.CompilerAsserts;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.*;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

@ExportLibrary(InteropLibrary.class)
public abstract class Function implements TruffleObject {

    public static final InteropLibrary INTEROP = InteropLibrary.getFactory().getUncached();

    private final String name;

    protected Function(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    protected static String expectString(Object argument, String errorMessage) throws UnsupportedTypeException {
        CompilerAsserts.neverPartOfCompilation();
        try {
            return INTEROP.asString(argument);
        } catch (UnsupportedMessageException e) {
            throw UnsupportedTypeException.create(new Object[]{argument}, errorMessage);
        }
    }

    public static int expectInt(Object number) throws UnsupportedTypeException {
        CompilerAsserts.neverPartOfCompilation();
        try {
            return INTEROP.asInt(number);
        } catch (UnsupportedMessageException e) {
            throw UnsupportedTypeException.create(new Object[]{number}, "expected integer number argument");
        }
    }

    protected static long expectLong(Object number, String message) throws UnsupportedTypeException {
        CompilerAsserts.neverPartOfCompilation();
        try {
            return INTEROP.asLong(number);
        } catch (UnsupportedMessageException e) {
            throw UnsupportedTypeException.create(new Object[]{number}, message);
        }
    }

    public static long expectLong(Object number) throws UnsupportedTypeException {
        return expectLong(number, "expected long number argument");
    }

    protected static int expectPositiveInt(Object number) throws UnsupportedTypeException {
        int value = expectInt(number);
        if (value < 0) {
            CompilerDirectives.transferToInterpreter();
            throw UnsupportedTypeException.create(new Object[]{number}, "expected positive int number argument");
        }
        return value;
    }

    public static long expectPositiveLong(Object number) throws UnsupportedTypeException {
        long value = expectLong(number);
        if (value < 0) {
            CompilerDirectives.transferToInterpreter();
            throw UnsupportedTypeException.create(new Object[]{number}, "expected positive long number argument");
        }
        return value;
    }

    public static void checkArgumentLength(Object[] arguments, int expected) throws ArityException {
        if (arguments.length != expected) {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(expected, arguments.length);
        }
    }

    public static void checkMinimumArgumentLength(Object[] arguments, int expected) throws ArityException {
        if (arguments.length < expected) {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(expected, arguments.length);
        }
    }

    public static Object expectObject(Object obj) throws UnsupportedTypeException {
        CompilerAsserts.neverPartOfCompilation();

        if (!INTEROP.hasMembers(obj)) {
            throw UnsupportedTypeException.create(new Object[]{obj}, "expected object");
        }

        return obj;
    }

    // InteropLibrary implementation

    @ExportMessage
    @SuppressWarnings("static-method")
    public final boolean isExecutable() {
        return true;
    }

    @ExportMessage
    public Object execute(Object[] arguments) throws ArityException, UnsupportedTypeException, UnsupportedMessageException {
        return call(arguments);
    }

    @SuppressWarnings("unused")
    protected Object call(Object[] arguments) throws ArityException, UnsupportedTypeException, UnsupportedMessageException {
        throw UnsupportedMessageException.create();
    }
}
