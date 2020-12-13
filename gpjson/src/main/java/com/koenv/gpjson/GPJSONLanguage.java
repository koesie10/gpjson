package com.koenv.gpjson;

import com.oracle.truffle.api.CallTarget;
import com.oracle.truffle.api.Truffle;
import com.oracle.truffle.api.TruffleLanguage;

import java.util.Objects;

@TruffleLanguage.Registration(id = GPJSONLanguage.ID, name = "gpjson", version = "0.1")
public class GPJSONLanguage extends TruffleLanguage<GPJSONContext> {
    public static final String ID = "gpjson";

    @Override
    protected GPJSONContext createContext(Env env) {
        if (!env.isNativeAccessAllowed()) {
            throw new GPJSONException("cannot create CUDA context without native access");
        }
        return new GPJSONContext(env);
    }

    @Override
    protected void disposeContext(GPJSONContext context) {
        context.dispose();
    }

    @Override
    protected CallTarget parse(ParsingRequest request) throws Exception {
        String source = request.getSource().getCharacters().toString();
        if (Objects.equals(source, "jsonpath")) {
            return Truffle.getRuntime().createCallTarget(new GPJSONRootNode(this));
        }

        throw new GPJSONParseError(request.getSource(), 1, 1, source.length(), false, "'jsonpath' expected");
    }

    public static GPJSONLanguage getCurrentLanguage() {
        return TruffleLanguage.getCurrentLanguage(GPJSONLanguage.class);
    }
}
