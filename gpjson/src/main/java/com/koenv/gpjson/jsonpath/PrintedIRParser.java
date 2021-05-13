package com.koenv.gpjson.jsonpath;

import java.util.Arrays;
import java.util.Scanner;
import java.util.stream.Collectors;

public class PrintedIRParser {
    public static JSONPathResult parse(Scanner scanner) {
        IRBuilder ir = new IRBuilder();

        int maxDepth = 0;

        while (scanner.hasNext()) {
            String line = scanner.nextLine().trim();

            String opcode = line.split(" ")[0];

            switch (opcode) {
                case "end":
                    ir.end();
                    break;
                case "store_result":
                case "storer":
                    ir.storeResult();
                    break;
                case "up":
                    ir.up();
                    break;
                case "down":
                    ir.down();

                    maxDepth = Math.max(ir.getCurrentLevel(), maxDepth);

                    break;
                case "property":
                    String[] propertyParts = line.split(" ");

                    assert(propertyParts.length >= 2);

                    String propertyName = Arrays.stream(propertyParts).skip(1).collect(Collectors.joining(" "));

                    ir.property(propertyName);

                    break;
                case "index":
                    String indexParts[] = line.split(" ");

                    assert(indexParts.length == 2);

                    ir.index(Integer.parseInt(indexParts[1]));
                    break;
            }
        }

        return new JSONPathResult(ir.getBuffer(), maxDepth, ir.getNumResultStores());
    }
}
