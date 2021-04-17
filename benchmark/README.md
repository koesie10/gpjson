# Benchmark

To run, execute your GraalVM node executable:

```shell
$GRAALVM_HOME/bin/node --polyglot --jvm --engine.TraceCompilation=true benchmark.js
```

`engine.TraceCompilation` is used to check whether all hot paths in the benchmark code are JIT-compiled.

The sequential benchmark can be run in the same way.

The dataset used for this benchmark should be in the `../datasets` directory, please see the README in that
directory for more information.
