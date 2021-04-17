package com.koenv.gpjson.gradlebuild.trufflenfi

import org.gradle.api.Project
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.provider.Property

abstract class TruffleNfiTestExtension(val project: Project) {
    abstract val platform: Property<String>

    abstract val architecture: Property<String>

    abstract val libSuffix: Property<String>

    abstract val nativesDirectory: DirectoryProperty

    init {
        val os = System.getProperty("os.name")
        val platformName = when {
            os.startsWith("Linux") -> "linux"
            os.startsWith("Mac OS X") || os.startsWith("Darwin") -> "darwin"
            else -> ""
        }

        platform.convention(platformName)

        val arch = System.getProperty("os.arch")
        val is64Bit = arch.contains("64") || arch.contains("armv8")

        val architectureName = when {
            arch.startsWith("aarch64") -> "aarch64"
            !arch.contains("arm") && is64Bit -> "amd64"
            else -> ""
        }

        architecture.convention(architectureName)

        libSuffix.convention(when(platform.get()) {
            "linux" -> ".so"
            "darwin" -> ".dylib"
            else -> ""
        })

        nativesDirectory.convention(project.layout.buildDirectory.dir("truffle-nfi-natives"))
    }
}