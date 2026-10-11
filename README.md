# NeonIntrinsics-Android

## Overview

An Android application that compares scalar C++ and Arm NEON implementations of an integer dot product. It demonstrates how to connect an Android Java activity to native processing through JNI, build the native library with CMake, and display the result and elapsed time in the application.

> This repository is retained as a historical technical example. It reflects the development environment and APIs used when the accompanying material was originally published in 2020.

## Technical focus

- Native C++ and Arm NEON SIMD intrinsics
- Integer vector loads, multiply-accumulate operations, and reduction
- Android NDK, CMake, and Gradle integration
- Java-to-native calls through JNI

## Workflow

`Java activity → JNI → generated input vectors → scalar / NEON dot product → result and timing text`

The application generates two 1,024-element vectors and runs each dot-product implementation repeatedly. The processing code is in [`native-lib.cpp`](app/src/main/cpp/native-lib.cpp); [`MainActivity.java`](app/src/main/java/com/example/neonintrinsics/MainActivity.java) loads the shared library and displays its output.

The elapsed times are illustrative measurements. Device, compiler optimization, build type, and measurement overhead affect the comparison; the repository does not establish a portable speedup.

## Known limitations

The native example needs correctness work before reuse: array cleanup uses scalar `delete`, and the NEON tail logic does not support arbitrary input lengths correctly. The checked-in tests are Android project-template tests, rather than algorithm-validation or benchmark tests.

## Getting started

### Historical toolchain

The checked-in project uses Android Gradle Plugin 3.6.2, Gradle 5.6.4, Android SDK 29, Build Tools 29.0.3, and CMake 3.10.2. The IDE configuration references JDK 8. The Android NDK version is not pinned.

1. Clone the repository and open its root directory in a compatible Android Studio environment.
2. Install Android SDK 29, Build Tools 29.0.3, the Android NDK, and CMake 3.10.2 through the SDK Manager.
3. For an Arm-only build, replace the existing ABI filter in [`app/build.gradle`](app/build.gradle) with:

   ```groovy
   ndk.abiFilters 'armeabi-v7a', 'arm64-v8a'
   ```

   The historical configuration also lists `x86`, but the native source includes `arm_neon.h` unconditionally and does not provide an x86 implementation.
4. Keep the existing CMake argument `-DANDROID_ARM_NEON=ON`, synchronize Gradle, and run the `app` configuration on an Android device with NEON support.

After configuring the historical toolchain and Arm ABIs, a command-line debug build is:

```sh
bash gradlew :app:assembleDebug
```

On Windows, use `gradlew.bat :app:assembleDebug`. The APK is generated under `app/build/outputs/apk/debug/`.

Current Android Studio releases may require additional setup for these older build tools and dependencies. This documentation update does not migrate the project or certify a build with a current Android toolchain.

## Related material

- [Arm developer guide: Neon Intrinsics — Getting Started on Android](https://documentation-service.arm.com/static/63cfdefbeae484354a6c9d43) — the public guide links to this repository and explains the native Android setup and dot-product example.
- [NeonIntrinsicsSamples](https://github.com/dawidborycki/NeonIntrinsicsSamples) — a related Android example covering signal thresholding and convolution.

## License

See [LICENSE](LICENSE) for the MIT license.
