import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar

plugins {
    id("com.github.johnrengelman.shadow")
    `java-library`
    id("com.koenv.gpjson.truffle-nfi-test")
}

val graalVmVersion: String by project

dependencies {
    api("org.graalvm.truffle", "truffle-api", graalVmVersion)
    annotationProcessor("org.graalvm.truffle", "truffle-dsl-processor", graalVmVersion)

    implementation("com.jayway.jsonpath", "json-path" , "2.5.0")

    testImplementation("org.graalvm.sdk", "graal-sdk", graalVmVersion)
    testRuntimeOnly("org.graalvm.truffle", "truffle-nfi", graalVmVersion)
    testImplementation("org.junit.jupiter", "junit-jupiter-api", "5.6.0")
    testImplementation("org.junit.jupiter", "junit-jupiter-params", "5.6.0")
    testRuntimeOnly("org.junit.jupiter", "junit-jupiter-engine")
}

tasks {
    named<ShadowJar>("shadowJar") {
        dependencies {
            exclude(dependency("org.graalvm.*:.*"))
        }

        // Relocate to prevent any version conflicts with anything else on the classpath
        relocate("com.jayway.jsonpath", "com.koenv.gpjson.thirdparty.com.jayway.jsonpath")
        relocate("net.minidev", "com.koenv.gpjson.thirdparty.net.minidev")
        relocate("org.objectweb", "com.koenv.gpjson.thirdparty.org.objectweb")
        relocate("org.slf4j", "com.koenv.gpjson.thirdparty.org.slf4j")
    }

    assemble {
        dependsOn(getByName("shadowJar"))
    }

    test {
        useJUnitPlatform()

        minHeapSize = "4096m"
        maxHeapSize = "16384m"
    }
}

tasks.register<Copy>("copyToGraalVM") {
    val shadowJar = tasks.getByName<ShadowJar>("shadowJar")

    dependsOn(shadowJar)

    from(shadowJar.outputs.files.files)

    if (!project.hasProperty("graalVMDirectory")) {
        throw IllegalArgumentException("Missing 'graalVMDirectory' property")
    }

    into("${project.property("graalVMDirectory")}/jre/languages/gpjson")
}
