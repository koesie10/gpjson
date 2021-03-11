package com.koenv.gpjson.kernel;

import com.koenv.gpjson.KernelTest;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.assertNotNull;

public class GPJSONKernelTest extends KernelTest {
    @ParameterizedTest
    @EnumSource(GPJSONKernel.class)
    public void compile(GPJSONKernel kernel) {
        assertNotNull(cudaRuntime.getKernel(kernel));
    }
}
