plugins {
    `java-gradle-plugin`
    `kotlin-dsl`
}

group = "com.koenv.gpjson.gradlebuild"
version = "0.1-SNAPSHOT"

repositories {
    gradlePluginPortal()
    mavenCentral()
}

gradlePlugin {
    plugins {
        create("truffleNfiTest") {
            id = "com.koenv.gpjson.truffle-nfi-test"
            implementationClass = "com.koenv.gpjson.gradlebuild.trufflenfi.TruffleNfiTestPlugin"
        }
    }
}

dependencies {
    testImplementation("junit", "junit", "4.13.1")
}