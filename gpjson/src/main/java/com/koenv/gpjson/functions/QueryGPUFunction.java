package com.koenv.gpjson.functions;

import com.koenv.gpjson.GPJSONContext;
import com.koenv.gpjson.GPJSONException;
import com.koenv.gpjson.gpu.CUDAMemcpyKind;
import com.koenv.gpjson.gpu.ManagedGPUPointer;
import com.koenv.gpjson.gpu.UnsafeHelper;
import com.koenv.gpjson.kernel.GPJSONKernel;
import com.koenv.gpjson.stages.NewlineIndex;
import com.koenv.gpjson.stages.StringIndex;
import com.koenv.gpjson.stages.LeveledBitmapsIndex;
import com.koenv.gpjson.util.FormatUtil;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.exception.AbstractTruffleException;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.TimeUnit;

public class QueryGPUFunction extends Function {
    private static final int FILE_READ_BUFFER = 1 << 20;

    private final GPJSONContext context;

    public QueryGPUFunction(GPJSONContext context) {
        super("queryGpu");
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

        context.getCudaRuntime().timings.start("queryGPU");

        checkArgumentLength(arguments, 2);
        String filename = expectString(arguments[0], "expected filename");
        String query = expectString(arguments[1], "expected query");

        Path file = Paths.get(filename);

        long size;
        try {
            size = Files.size(file);
        } catch (IOException e) {
            throw new GPJSONException("Failed to get size of file", e, AbstractTruffleException.UNLIMITED_STACK_TRACE, null);
        }

        try (ManagedGPUPointer fileMemory = context.getCudaRuntime().allocateUnmanagedMemory(size)) {
            readFile(fileMemory, file, size);

            end = System.nanoTime();
            long duration = end - start;
            double durationSeconds = duration / (double) TimeUnit.SECONDS.toNanos(1);
            double speed = size / durationSeconds;

            System.out.printf("Reading file done in %dms, %s/second%n", TimeUnit.NANOSECONDS.toMillis(duration), FormatUtil.humanReadableByteCountSI((long) speed));

            start = System.nanoTime();

            NewlineIndex newlineIndexCreator = new NewlineIndex(context.getCudaRuntime(), fileMemory);
            long[] newlineIndex = newlineIndexCreator.create();

            end = System.nanoTime();
            duration = end - start;
            durationSeconds = duration / (double) TimeUnit.SECONDS.toNanos(1);
            speed = size / durationSeconds;

            System.out.printf("Creating newline index done in %dms, %s/second%n", TimeUnit.NANOSECONDS.toMillis(duration), FormatUtil.humanReadableByteCountSI((long) speed));
            // System.out.println(Arrays.toString(newlineIndex));
            System.out.println(newlineIndex.length);

            start = System.nanoTime();

            StringIndex stringIndexCreator = new StringIndex(context.getCudaRuntime(), fileMemory);

            try (ManagedGPUPointer stringIndex = stringIndexCreator.create()) {
                end = System.nanoTime();
                duration = end - start;
                durationSeconds = duration / (double) TimeUnit.SECONDS.toNanos(1);
                speed = size / durationSeconds;

                System.out.printf("Creating string index done in %dms, %s/second%n", TimeUnit.NANOSECONDS.toMillis(duration), FormatUtil.humanReadableByteCountSI((long) speed));

                start = System.nanoTime();

                LeveledBitmapsIndex leveledBitmapsIndexCreator = new LeveledBitmapsIndex(context.getCudaRuntime(), fileMemory, stringIndex);

                try (ManagedGPUPointer leveledBitmapIndex = leveledBitmapsIndexCreator.create()) {
                    end = System.nanoTime();
                    duration = end - start;
                    durationSeconds = duration / (double) TimeUnit.SECONDS.toNanos(1);
                    speed = size / durationSeconds;

                    System.out.printf("Creating structural index done in %dms, %s/second%n", TimeUnit.NANOSECONDS.toMillis(duration), FormatUtil.humanReadableByteCountSI((long) speed));

                    // TODO: Use structural index
                }
            }
        }

        // queryGPU
        context.getCudaRuntime().timings.end();

        return context.getCudaRuntime().timings.export();
    }

    private void readFile(ManagedGPUPointer memory, Path file, long expectedSize) {
        context.getCudaRuntime().timings.start("readFile");

        ByteBuffer buffer = ByteBuffer.allocateDirect(FILE_READ_BUFFER);
        UnsafeHelper.ByteArray byteArray = UnsafeHelper.createByteArray(buffer);

        try (FileChannel channel = FileChannel.open(file);) {
            if (channel.size() != expectedSize) {
                throw new GPJSONException("Size of file has changed while reading");
            }

            long offset = 0;

            while (true) {
                buffer.clear();

                int numBytes = channel.read(buffer);
                if (numBytes <= 0) {
                    break;
                }

                if (offset + numBytes > expectedSize) {
                    throw new GPJSONException("Size of file has changed while reading");
                }

                context.getCudaRuntime().cudaMemcpy(memory.getPointer().getRawPointer() + offset, byteArray.getAddress(), numBytes, CUDAMemcpyKind.HOST_TO_DEVICE);

                offset += numBytes;
            }
        } catch (IOException e) {
            throw new GPJSONException("Failed to open file", e, AbstractTruffleException.UNLIMITED_STACK_TRACE, null);
        } finally {
            context.getCudaRuntime().timings.end();
        }
    }
}
