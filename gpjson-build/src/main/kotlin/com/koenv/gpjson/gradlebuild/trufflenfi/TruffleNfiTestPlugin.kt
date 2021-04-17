package com.koenv.gpjson.gradlebuild.trufflenfi

import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.api.plugins.JavaPlugin
import org.gradle.api.tasks.Copy
import org.gradle.api.tasks.testing.Test
import org.gradle.kotlin.dsl.create
import org.gradle.kotlin.dsl.getByName
import org.gradle.kotlin.dsl.register
import java.io.File

class TruffleNfiTestPlugin : Plugin<Project> {
    override fun apply(project: Project) {
        val extension = project.extensions.create<TruffleNfiTestExtension>("truffleNfiTest", project)

        val unzipTruffleNfiNatives = project.tasks.register<Copy>("unzipTruffleNfiNatives") {
            inputs.property("platform", extension.platform.get())
            inputs.property("architecture", extension.architecture.get())

            val nativesFile = project.configurations
                .getByName(JavaPlugin.TEST_RUNTIME_CLASSPATH_CONFIGURATION_NAME)
                .resolvedConfiguration.resolvedArtifacts
                .find { it.moduleVersion.id.group == "org.graalvm.truffle" && it.moduleVersion.id.name == "truffle-nfi-native-${extension.platform.get()}-${extension.architecture.get()}" }
                ?: throw UnsupportedOperationException("No Truffle natives found for platform ${extension.platform.get()}-${extension.architecture.get()}")

            from(project.tarTree(nativesFile.file)) {
                include("bin/*")
                eachFile {
                    // Flatten directories
                    path = name
                }
            }
            into(extension.nativesDirectory.get())
        }

        project.afterEvaluate {
            project.tasks.getByName<Test>("test") {
                dependsOn(unzipTruffleNfiNatives)

                inputs.property("libSuffix", extension.libSuffix.get())

                systemProperty("truffle.nfi.library", File(extension.nativesDirectory.get().asFile, "libtrufflenfi${extension.libSuffix.get()}").absolutePath)
            }
        }
    }
}