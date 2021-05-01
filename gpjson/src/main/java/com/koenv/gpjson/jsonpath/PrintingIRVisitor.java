package com.koenv.gpjson.jsonpath;

import java.io.PrintStream;

public class PrintingIRVisitor implements IRVisitor {
    private final PrintStream printStream;

    public PrintingIRVisitor(PrintStream printStream) {
        this.printStream = printStream;
    }

    @Override
    public void visitProperty(String name) {
        this.printStream.printf("property %s%n", name);
    }

    @Override
    public void visitIndex(int index) {
        this.printStream.printf("index %d%n", index);
    }

    @Override
    public void visitEnd() {
        this.printStream.println("end");
    }
}
