plugins {
    id("com.github.johnrengelman.shadow")
    java
    application
}

val graalVmVersion: String by project

application {
    mainClass.set("com.koenv.gpjson.launcher.Launcher")
    // Deprecated, but required for the shadow plugin
    mainClassName = "com.koenv.gpjson.launcher.Launcher"
}

dependencies {
    implementation(project(":gpjson"))
    implementation("org.graalvm.sdk", "graal-sdk", graalVmVersion)
    implementation("org.graalvm.js", "js", graalVmVersion)

    testImplementation("org.junit.jupiter", "junit-jupiter-api", "5.6.0")
    testRuntimeOnly("org.junit.jupiter", "junit-jupiter-engine")
}

tasks {
    test {
        useJUnitPlatform()
    }
}
