package com.koenv.gpjson.functions;

import com.jayway.jsonpath.*;
import com.koenv.gpjson.GPJSONContext;
import com.koenv.gpjson.GPJSONException;
import com.koenv.gpjson.debug.GPUUtils;
import com.koenv.gpjson.gpu.*;
import com.koenv.gpjson.jsonpath.*;
import com.koenv.gpjson.kernel.GPJSONKernel;
import com.koenv.gpjson.result.FallbackResultArray;
import com.koenv.gpjson.result.ResultArray;
import com.koenv.gpjson.stages.CombinedIndex;
import com.koenv.gpjson.stages.CombinedIndexResult;
import com.koenv.gpjson.stages.LeveledBitmapsIndex;
import com.koenv.gpjson.util.FormatUtil;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.exception.AbstractTruffleException;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnknownIdentifierException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
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
        String query = expectString(arguments[1], "expected query");

        QueryOptions queryOptions = new QueryOptions();

        if (arguments.length > 2) {
            try {
                queryOptions.from(INTEROP, expectObject(arguments[2]));
            } catch (UnsupportedMessageException | UnknownIdentifierException e) {
                throw new GPJSONException("Invalid options", e, AbstractTruffleException.UNLIMITED_STACK_TRACE, null);
            }
        }

        Path file = Paths.get(filename);

        JSONPathResult compiledQuery;
        try {
            compiledQuery = new JSONPathParser(new JSONPathScanner(query)).compile();
        } catch (UnsupportedJSONPathException e) {
            if (queryOptions.disableFallback) {
                throw new GPJSONException("Unsupported JSON path (fallback disabled)", e, AbstractTruffleException.UNLIMITED_STACK_TRACE, null);
            }

            // If the path is unsupported, we'll fall back to a Java JsonPath implementation
            return fallbackQuery(file, query);
        } catch (JSONPathException e) {
            throw new GPJSONException("Unsupported JSON path", e, AbstractTruffleException.UNLIMITED_STACK_TRACE, null);
        }
        ByteBuffer compiledQueryBuffer = compiledQuery.getIr().toByteBuffer();

        long size;
        try {
            size = Files.size(file);
        } catch (IOException e) {
            throw new GPJSONException("Failed to get size of file", e, AbstractTruffleException.UNLIMITED_STACK_TRACE, null);
        }

        long[] returnValue;
        long numberOfReturnValues;

        try (
                ManagedGPUPointer fileMemory = context.getCudaRuntime().allocateUnmanagedMemory(size);
                ManagedGPUPointer queryMemory = context.getCudaRuntime().allocateUnmanagedMemory(compiledQueryBuffer.capacity())
        ) {
            start = System.nanoTime();
            readFile(fileMemory, file, size);

            end = System.nanoTime();
            long duration = end - start;
            double durationSeconds = duration / (double) TimeUnit.SECONDS.toNanos(1);
            double speed = size / durationSeconds;

            System.out.printf("Reading file done in %dms, %s/second%n", TimeUnit.NANOSECONDS.toMillis(duration), FormatUtil.humanReadableByteCountSI((long) speed));

            queryMemory.loadFrom(compiledQueryBuffer);

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

                LeveledBitmapsIndex leveledBitmapsIndexCreator = new LeveledBitmapsIndex(context.getCudaRuntime(), fileMemory, stringIndex, (byte) compiledQuery.getMaxDepth());

                try (ManagedGPUPointer leveledBitmapIndex = leveledBitmapsIndexCreator.create()) {
                    end = System.nanoTime();
                    duration = end - start;
                    durationSeconds = duration / (double) TimeUnit.SECONDS.toNanos(1);
                    speed = size / durationSeconds;

                    System.out.printf("Creating leveled bitmaps index done in %dms, %s/second%n", TimeUnit.NANOSECONDS.toMillis(duration), FormatUtil.humanReadableByteCountSI((long) speed));

                    try (ManagedGPUPointer result = context.getCudaRuntime().allocateUnmanagedMemory(newlineIndex.numberOfElements() * 2, Type.SINT64)) {
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
                        kernelArguments.add(UnsafeHelper.createInteger32Object(compiledQueryBuffer.capacity()));

                        kernelArguments.add(UnsafeHelper.createPointerObject(result));

                        start = System.nanoTime();

                        kernel.execute(new Dim3(8), new Dim3(1024), 0, 0, kernelArguments);

                        end = System.nanoTime();
                        duration = end - start;
                        durationSeconds = duration / (double) TimeUnit.SECONDS.toNanos(1);
                        speed = size / durationSeconds;

                        System.out.printf("Finding values done in %dms, %s/second%n", TimeUnit.NANOSECONDS.toMillis(duration), FormatUtil.humanReadableByteCountSI((long) speed));

                        numberOfReturnValues = newlineIndex.numberOfElements();
                        returnValue = GPUUtils.readLongs(result);
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

        return new ResultArray(file, numberOfReturnValues, returnValue, mappedBuffer);
    }

    private Object fallbackQuery(Path file, String query) {
        context.getCudaRuntime().timings.start("fallbackQuery");

        Configuration conf = Configuration.defaultConfiguration()
                .addOptions(Option.ALWAYS_RETURN_LIST, Option.ALWAYS_RETURN_LIST);

        ParseContext parseContext = JsonPath.using(conf);

        JsonPath path = JsonPath.compile(query);

        try (Stream<String> lines = Files.lines(file, StandardCharsets.UTF_8)) {
            return new FallbackResultArray(lines.parallel().map(line -> {
                try {
                    return Collections.singletonList(parseContext.parse(line).read(path).toString());
                } catch (PathNotFoundException e) {
                    return new ArrayList<String>();
                }
            }).collect(Collectors.toList()));
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
