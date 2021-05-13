package com.koenv.gpjson.functions;

import com.jayway.jsonpath.*;
import com.koenv.gpjson.GPJSONContext;
import com.koenv.gpjson.GPJSONException;
import com.koenv.gpjson.debug.GPUUtils;
import com.koenv.gpjson.gpu.*;
import com.koenv.gpjson.jsonpath.*;
import com.koenv.gpjson.kernel.GPJSONKernel;
import com.koenv.gpjson.result.FallbackResult;
import com.koenv.gpjson.result.Result;
import com.koenv.gpjson.stages.CombinedIndex;
import com.koenv.gpjson.stages.CombinedIndexResult;
import com.koenv.gpjson.stages.LeveledBitmapsIndex;
import com.koenv.gpjson.util.FormatUtil;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.exception.AbstractTruffleException;
import com.oracle.truffle.api.interop.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class QueryFunction extends Function {
    private final GPJSONContext context;

    public QueryFunction(GPJSONContext context) {
        super("query");
        this.context = context;
    }

    @Override
    @CompilerDirectives.TruffleBoundary
    public Object call(Object[] arguments) throws UnsupportedTypeException, ArityException {
        long start = System.nanoTime();

        context.getCudaRuntime().timings.start("compile_kernels");
        for (GPJSONKernel kernel : GPJSONKernel.values()) {
            context.getCudaRuntime().getKernel(kernel);
        }
        context.getCudaRuntime().timings.end();

        long end = System.nanoTime();
        System.out.printf("Compiling kernels done in %dms%n", TimeUnit.NANOSECONDS.toMillis(end - start));

        context.getCudaRuntime().timings.start("query");

        checkMinimumArgumentLength(arguments, 2);
        String filename = expectString(arguments[0], "expected filename");

        List<String> queries;

        try {
            if (INTEROP.isString(arguments[1])) {
                queries = Collections.singletonList(INTEROP.asString(arguments[1]));
            } else if (INTEROP.hasArrayElements(arguments[1])) {
                long queriesSize = INTEROP.getArraySize(arguments[1]);
                // Arbitrary limit
                if (queriesSize > 255) {
                    throw new GPJSONException("Maximum size of queries is 255", null, AbstractTruffleException.UNLIMITED_STACK_TRACE, null);
                }

                queries = new ArrayList<>((int) queriesSize);

                for (long i = 0; i < queriesSize; i++) {
                    Object item = INTEROP.readArrayElement(arguments[1], i);

                    try {
                        queries.add(INTEROP.asString(item));
                    } catch (UnsupportedMessageException e) {
                        throw UnsupportedTypeException.create(new Object[]{item}, "expected queries to be a string or string array");
                    }
                }
            } else {
                throw UnsupportedTypeException.create(new Object[]{arguments[1]}, "expected queries to be a string or string array");
            }
        } catch (UnsupportedMessageException | InvalidArrayIndexException e) {
            throw UnsupportedTypeException.create(new Object[]{arguments[1]}, "expected queries to be a string or string array");
        }

        if (queries.size() < 1) {
            throw UnsupportedTypeException.create(new Object[]{arguments[1]}, "expected queries to contain at least one query");
        }

        QueryOptions queryOptions = new QueryOptions();

        if (arguments.length > 2) {
            try {
                queryOptions.from(INTEROP, expectObject(arguments[2]));
            } catch (UnsupportedMessageException | UnknownIdentifierException e) {
                throw new GPJSONException("Invalid options", e, AbstractTruffleException.UNLIMITED_STACK_TRACE, null);
            }
        }

        Path file = Paths.get(filename);

        List<JSONPathResult> compiledQueries = new ArrayList<>(queries.size());

        for (int i = 0; i < queries.size(); i++) {
            String query = queries.get(i);

            try {
                JSONPathResult compiledQuery = new JSONPathParser(new JSONPathScanner(query)).compile();

                IRVisitor.accept(compiledQuery.getIr().toReadable(), new PrintingIRVisitor(System.out));

                compiledQueries.add(compiledQuery);
            } catch (UnsupportedJSONPathException e) {
                if (queryOptions.disableFallback) {
                    throw new GPJSONException("Unsupported JSON path (" + i + ", fallback disabled)", e, AbstractTruffleException.UNLIMITED_STACK_TRACE, null);
                }

                // If the path is unsupported, we'll fall back to a Java JsonPath implementation
                return fallbackQuery(file, queries);
            } catch (JSONPathException e) {
                throw new GPJSONException("Unsupported JSON path (" + i + ")" , e, AbstractTruffleException.UNLIMITED_STACK_TRACE, null);
            }
        }

        int maxDepth = compiledQueries.stream().mapToInt(JSONPathResult::getMaxDepth).max().getAsInt();
        int maxQuerySize = compiledQueries.stream().mapToInt(query -> query.getIr().size()).max().getAsInt();
        int maxNumResults = compiledQueries.stream().mapToInt(JSONPathResult::getNumResults).max().getAsInt();

        long size;
        try {
            size = Files.size(file);
        } catch (IOException e) {
            throw new GPJSONException("Failed to get size of file", e, AbstractTruffleException.UNLIMITED_STACK_TRACE, null);
        }

        long[][] returnValue = new long[compiledQueries.size()][];
        long numberOfLines;

        try (
                ManagedGPUPointer fileMemory = context.getCudaRuntime().allocateUnmanagedMemory(size)
        ) {
            start = System.nanoTime();
            readFile(fileMemory, file, size);

            end = System.nanoTime();
            long duration = end - start;
            double durationSeconds = duration / (double) TimeUnit.SECONDS.toNanos(1);
            double speed = size / durationSeconds;

            System.out.printf("Reading file done in %dms, %s/second%n", TimeUnit.NANOSECONDS.toMillis(duration), FormatUtil.humanReadableByteCountSI((long) speed));

            start = System.nanoTime();

            CombinedIndex combinedIndexCreator = new CombinedIndex(context.getCudaRuntime(), fileMemory);
            try (CombinedIndexResult combinedIndexResult = combinedIndexCreator.create()) {
                ManagedGPUPointer stringIndex = combinedIndexResult.stringIndex;
                ManagedGPUPointer newlineIndex = combinedIndexResult.newlineIndex;

                end = System.nanoTime();
                duration = end - start;
                durationSeconds = duration / (double) TimeUnit.SECONDS.toNanos(1);
                speed = size / durationSeconds;

                System.out.printf("Creating newline and string index done in %dms, %s/second%n", TimeUnit.NANOSECONDS.toMillis(duration), FormatUtil.humanReadableByteCountSI((long) speed));

                start = System.nanoTime();

                LeveledBitmapsIndex leveledBitmapsIndexCreator = new LeveledBitmapsIndex(context.getCudaRuntime(), fileMemory, stringIndex, (byte) maxDepth);

                try (ManagedGPUPointer leveledBitmapIndex = leveledBitmapsIndexCreator.create()) {
                    end = System.nanoTime();
                    duration = end - start;
                    durationSeconds = duration / (double) TimeUnit.SECONDS.toNanos(1);
                    speed = size / durationSeconds;

                    System.out.printf("Creating leveled bitmaps index done in %dms, %s/second%n", TimeUnit.NANOSECONDS.toMillis(duration), FormatUtil.humanReadableByteCountSI((long) speed));

                    try (
                            ManagedGPUPointer result = context.getCudaRuntime().allocateUnmanagedMemory(newlineIndex.numberOfElements() * 2 * maxNumResults, Type.SINT64);
                            ManagedGPUPointer queryMemory = context.getCudaRuntime().allocateUnmanagedMemory(maxQuerySize)
                    ) {
                        context.getCudaRuntime().cudaMemset(result.getPointer().getRawPointer(), -1, result.size());

                        numberOfLines = newlineIndex.numberOfElements();

                        for (int i = 0; i < compiledQueries.size(); i++) {
                            JSONPathResult compiledQuery = compiledQueries.get(i);

                            ByteBuffer compiledQueryBuffer = compiledQuery.getIr().toByteBuffer();
                            queryMemory.loadFrom(compiledQueryBuffer);

                            Kernel kernel = context.getCudaRuntime().getKernel(GPJSONKernel.FIND_VALUE);

                            List<UnsafeHelper.MemoryObject> kernelArguments = new ArrayList<>();
                            kernelArguments.add(UnsafeHelper.createPointerObject(fileMemory));
                            kernelArguments.add(UnsafeHelper.createInteger64Object(fileMemory.numberOfElements()));
                            kernelArguments.add(UnsafeHelper.createPointerObject(newlineIndex));
                            kernelArguments.add(UnsafeHelper.createInteger64Object(newlineIndex.numberOfElements()));
                            kernelArguments.add(UnsafeHelper.createPointerObject(stringIndex));
                            kernelArguments.add(UnsafeHelper.createPointerObject(leveledBitmapIndex));
                            kernelArguments.add(UnsafeHelper.createInteger64Object(leveledBitmapIndex.numberOfElements()));

                            long levelSize = (fileMemory.size() + 64 - 1) / 64;

                            kernelArguments.add(UnsafeHelper.createInteger64Object(levelSize));

                            kernelArguments.add(UnsafeHelper.createPointerObject(queryMemory));
                            kernelArguments.add(UnsafeHelper.createInteger32Object(maxNumResults));

                            kernelArguments.add(UnsafeHelper.createPointerObject(result));

                            start = System.nanoTime();

                            kernel.execute(new Dim3(8), new Dim3(1024), 0, 0, kernelArguments);

                            end = System.nanoTime();
                            duration = end - start;
                            durationSeconds = duration / (double) TimeUnit.SECONDS.toNanos(1);
                            speed = size / durationSeconds;

                            System.out.printf("Finding values done in %dms, %s/second%n", TimeUnit.NANOSECONDS.toMillis(duration), FormatUtil.humanReadableByteCountSI((long) speed));

                            returnValue[i] = GPUUtils.readLongs(result);
                        }
                    }
                }
            }
        }

        context.getCudaRuntime().timings.start("openFileChannel");

        MappedByteBuffer mappedBuffer;
        try {
            FileChannel channel = FileChannel.open(file);

            mappedBuffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
            mappedBuffer.load();
        } catch (IOException e) {
            throw new GPJSONException("Failed to open file", e, AbstractTruffleException.UNLIMITED_STACK_TRACE, null);
        }

        context.getCudaRuntime().timings.end();

        // query
        context.getCudaRuntime().timings.end();

        System.out.println(Arrays.deepToString(returnValue));

        return new Result(file, compiledQueries.size(), numberOfLines, maxNumResults, returnValue, mappedBuffer);
    }

    private Object fallbackQuery(Path file, List<String> queries) {
        context.getCudaRuntime().timings.start("fallbackQuery");

        Configuration conf = Configuration.defaultConfiguration()
                .addOptions(Option.ALWAYS_RETURN_LIST, Option.ALWAYS_RETURN_LIST);

        ParseContext parseContext = JsonPath.using(conf);

        List<JsonPath> paths = queries.stream().map(JsonPath::compile).collect(Collectors.toList());

        try (Stream<String> lines = Files.lines(file, StandardCharsets.UTF_8)) {
            List<List<List<String>>> results = lines.parallel().map(line -> {
                List<List<String>> pathResults = new ArrayList<>(paths.size());

                for (JsonPath path : paths) {
                    try {
                        pathResults.add(Collections.singletonList(parseContext.parse(line).read(path).toString()));
                    } catch (PathNotFoundException e) {
                        pathResults.add(Collections.emptyList());
                    }
                }

                return pathResults;
            }).collect(Collectors.toList());

            // The order now is line -> query -> results, while we want
            // it to be query -> line -> results

            List<List<List<String>>> transformedResults = new ArrayList<>(queries.size());
            for (int i = 0; i < queries.size(); i++) {
                transformedResults.add(new ArrayList<>());
            }

            for (List<List<String>> resultItem : results) {
                // resultItem is path -> results
                for (int i = 0; i < queries.size(); i++) {
                    transformedResults.get(i).add(resultItem.get(i));
                }
            }

            return new FallbackResult(transformedResults);
        } catch (IOException e) {
            throw new GPJSONException("Failed to read file", e, AbstractTruffleException.UNLIMITED_STACK_TRACE, null);
        } finally {
            // fallbackQuery
            context.getCudaRuntime().timings.end();
        }
    }

    private void readFile(ManagedGPUPointer memory, Path file, long expectedSize) {
        context.getCudaRuntime().timings.start("readFile");

        try (FileChannel channel = FileChannel.open(file)) {
            if (channel.size() != expectedSize) {
                throw new GPJSONException("Size of file has changed while reading");
            }

            MappedByteBuffer mappedBuffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
            mappedBuffer.load();

            UnsafeHelper.ByteArray byteArray = UnsafeHelper.createByteArray(mappedBuffer);

            context.getCudaRuntime().cudaMemcpy(memory.getPointer().getRawPointer(), byteArray.getAddress(), channel.size(), CUDAMemcpyKind.HOST_TO_DEVICE);
        } catch (IOException e) {
            throw new GPJSONException("Failed to open file", e, AbstractTruffleException.UNLIMITED_STACK_TRACE, null);
        } finally {
            context.getCudaRuntime().timings.end();
        }
    }
}
