package com.koenv.gpjson;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.TruffleLanguage;
import com.oracle.truffle.api.frame.VirtualFrame;
import com.oracle.truffle.api.nodes.RootNode;

public class GPJSONRootNode extends RootNode {
    @CompilerDirectives.CompilationFinal
    private TruffleLanguage.ContextReference<GPJSONContext> reference;

    protected GPJSONRootNode(TruffleLanguage<?> language) {
        super(language);
    }

    @Override
    public Object execute(VirtualFrame frame) {
        if (reference == null) {
            CompilerDirectives.transferToInterpreterAndInvalidate();
            this.reference = lookupContextReference(GPJSONLanguage.class);
        }

        GPJSONContext context = this.reference.get();
        return context.getRoot();
    }
}
