plugins {
    id("com.github.johnrengelman.shadow") version("6.1.0") apply(false)
}

subprojects {
    group = "com.koenv"
    version = "0.1-SNAPSHOT"

    repositories {
        mavenCentral()
    }
}
