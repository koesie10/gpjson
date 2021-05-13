package com.koenv.gpjson.jsonpath;

import java.io.PrintStream;

public class PrintingIRVisitor implements IRVisitor {
    private final PrintStream printStream;
    private int currentLevel = 0;

    public PrintingIRVisitor(PrintStream printStream) {
        this.printStream = printStream;
    }

    @Override
    public void visitProperty(String name) {
        this.printlnf("property %s%n", name);
    }

    @Override
    public void visitIndex(int index) {
        this.printlnf("index %d%n", index);
    }

    @Override
    public void visitDown() {
        this.println("down");
        currentLevel++;
    }

    @Override
    public void visitUp() {
        this.println("up");
        currentLevel--;
    }

    @Override
    public void visitStoreResult() {
        this.println("storer");
    }

    @Override
    public void visitEnd() {
        this.println("end");
    }

    private void println(String str) {
        this.printStream.println(indent() + str);
    }

    private void printlnf(String format, Object... args) {
        this.printStream.printf(indent() + format, args);
    }

    private String indent() {
        return new String(new char[currentLevel]).replace("\0", "  ");
    }
}
