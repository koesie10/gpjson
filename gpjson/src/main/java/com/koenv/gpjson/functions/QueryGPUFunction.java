package com.koenv.gpjson.functions;

import com.koenv.gpjson.GPJSONContext;
import com.koenv.gpjson.GPJSONException;
import com.koenv.gpjson.gpu.*;
import com.koenv.gpjson.kernel.GPJSONKernel;
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
import java.util.ArrayList;
import java.util.List;
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

        try (ManagedGPUMemory memory = context.getCudaRuntime().allocateMemory(size)) {
            readFile(memory, file, size);

            long end = System.nanoTime();
            long duration = end - start;
            double durationSeconds = duration / (double) TimeUnit.SECONDS.toNanos(1);
            double speed = size / durationSeconds;

            System.out.printf("Reading file done in %dms, %s/second%n", TimeUnit.NANOSECONDS.toMillis(duration), FormatUtil.humanReadableByteCountSI((long) speed));

            discoverStructure(memory, (int) size);
        }

        return "test";
    }

    private void discoverStructure(ManagedGPUMemory memory, int size) {
        Kernel kernel = context.getCudaRuntime().getKernel(GPJSONKernel.DISCOVER_STRUCTURE);

        List<UnsafeHelper.MemoryObject> arguments = new ArrayList<>();
        arguments.add(UnsafeHelper.createPointerObject(memory.getView().getStartAddress()));
        arguments.add(UnsafeHelper.createInteger64Object(size));

        kernel.execute(new Dim3(1), new Dim3(128), 0, 0, arguments);
    }

    private void readFile(ManagedGPUMemory memory, Path file, long expectedSize) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(FILE_READ_BUFFER);

        try (FileChannel channel = FileChannel.open(file);) {
            if (channel.size() != expectedSize) {
                throw new GPJSONException("Size of file has changed while reading");
            }

            UnsafeHelper.ByteArray byteArray = UnsafeHelper.createByteArray(FILE_READ_BUFFER);

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

                for (int i = 0; i < numBytes; i++) {
                    byteArray.setValueAt(i, buffer.get(i));
                }

                context.getCudaRuntime().cudaMemcpy(memory.getView().getStartAddress() + offset, byteArray.getAddress(), numBytes, CUDAMemcpyKind.HOST_TO_DEVICE);

                offset += numBytes;
            }
        } catch (IOException e) {
            throw new GPJSONException("Failed to open file", e, AbstractTruffleException.UNLIMITED_STACK_TRACE, null);
        }
    }
}
