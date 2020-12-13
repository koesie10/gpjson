package com.koenv.gpjson;

import com.oracle.truffle.api.TruffleLanguage;

public class GPJSONContext {
    private final TruffleLanguage.Env env;
    private final GPJSONLibrary root;

    public GPJSONContext(TruffleLanguage.Env env) {
        this.env = env;
        this.root = new GPJSONLibrary();
    }

    public GPJSONLibrary getRoot() {
        return root;
    }

    public void dispose() {

    }
}
