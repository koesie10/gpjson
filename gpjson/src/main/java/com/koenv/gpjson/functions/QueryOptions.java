package com.koenv.gpjson.functions;

import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.UnknownIdentifierException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;

public class QueryOptions {
    public boolean disableFallback;

    public void from(InteropLibrary interop, Object obj) throws UnsupportedMessageException, UnknownIdentifierException {
        if (interop.isMemberReadable(obj, "disable_fallback")) {
            disableFallback = interop.asBoolean(interop.readMember(obj, "disable_fallback"));
        }
    }
}
