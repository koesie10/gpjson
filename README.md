# GPJSON

GPJSON is a GPU-based JSON data processing system which can be used to execute
JSONPath queries on newline-delimited JSON datasets. It has support for a subset
of JSONPath queries and can be used from any GraalVM language.

## Build and installation

GPJSON uses [Gradle](https://gradle.org/) as a build system. There is no need to install
Gradle, the Gradle wrapper can be used to build GPJSON.

The only requirements for running GPJSON are GraalVM and CUDA (and a supported GPU). It has been tested
on GraalVM CE 21.0.0.2 Java 8, downloadable [here](https://github.com/graalvm/graalvm-ce-builds/releases/tag/vm-21.0.0.2).
It has only been tested on Linux using CUDA 11.2, but should work on other OS when
[CUDARuntime.java](https://github.com/koesie10/gpjson/blob/master/gpjson/src/main/java/com/koenv/gpjson/gpu/CUDARuntime.java)
is adapted.

To build the project, execute:

```shell
./gradlew build
```

This will generate two files in `gpjson/build/libs`:

* `gpjson-0.1-SNAPSHOT.jar`: Contains the compiled code of GPJSON, without any dependencies. This will not run without
including GPJSON's dependencies on the classpath.
* `gpjson-0.1-SNAPSHOT-all.jar`: Contains the compiled code of GPJSON, including all dependencies. This will run
without any other JAR files required on the classpath.
  
To use GPJSON in GraalVM, copy the `gpjson-0.1-SNAPSHOT-all.jar` file to `$GRAALVM_HOME/jre/languages/gpjson`. This can
also be done automatically by setting the `graalVMDirectory` project property. For example, if you have set the `GRAALVM_HOME`
environment variable, use:

```shell
./gradlew copyToGraalVM -PgraalVMDirectory=$GRAALVM_HOME
```

## Usage

Create a file `gpjson.js`:

```js
const gpjson = Polyglot.eval('gpjson', 'jsonpath');
const result = gpjson.query('dataset.json', '$.user.lang');
```

Then, run:

```shell
$GRAALVM_HOME/bin/node --polyglot --jvm gpjson.js
```

## License

GPJSON is licensed under the BSD 3-Clause license available [here](/LICENSE).

Parts of the source code are from grCUDA and are licensed under the BSD 3-Clause license available
[here](https://github.com/NVIDIA/grcuda/blob/b1f531a844d2906080ab4c04a33e99fe6e04089c/LICENSE).

GPJSON also depends on Truffle APIs licensed under the Universal Permissive
License (UPL), Version 1.0 (https://opensource.org/licenses/UPL).
