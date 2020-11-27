#include <arm_neon.h>
#include <ctime>
#include <jni.h>
#include <stdio.h>

class Timer
{
private:
    timespec beg;
    timespec end;

public:
    Timer() { clock_gettime(CLOCK_REALTIME, &beg); }

    double elapsedMs() {
        clock_gettime(CLOCK_REALTIME, &end);
        return (end.tv_sec - beg.tv_sec) * 1000.f +
            (end.tv_nsec - beg.tv_nsec) / 1000000.f;
    }

    void reset() { clock_gettime(CLOCK_REALTIME, &beg); }
};

static short* generateRamp(short startValue, short len)
{
    short* ramp = new short[len];

    for (short i = 0; i < len; i++)
    {
        ramp[i] = startValue + i;
    }

    return ramp;
}

int dotProduct(short* vector1, short* vector2, short len)
{
    int result = 0;

    for (short i = 0; i < len; i++)
    {
        result += vector1[i] * vector2[i];
    }

    return result;
}

int dotProductNeon(short* vector1, short* vector2, short len)
{
    const short transferSize = 4;
    short segments = len / transferSize;

    // 4-element vector of zeros
    int32x4_t partialSumsNeon = vdupq_n_s32(0);

    // Main loop (note that loop index goes through segments). Unroll 2-wide
    for (int i = 0; i < segments; ++i)
    {
        // Load vector elements to registers
        int16x4_t v1 = vld1_s16(vector1);
        int16x4_t v2 = vld1_s16(vector2);

        partialSumsNeon = vmlal_s16(partialSumsNeon, v1, v2);

        vector1 += 4;
        vector2 += 4;
    }

    // Store partial sums
    int partialSums[transferSize];
    vst1q_s32(partialSums, partialSumsNeon);

    // Sum up partial sums
    int result = 0;
    for(int i = 0; i < transferSize; i++)
    {
        result += partialSums[i];
    }

    return result;
}

int dotProductNeon2(short* vector1, short* vector2, short len)
{
    const short transferSize = 4;
    short segments = len / (2 * transferSize);

    // 4-element vector of zeros
    int32x4_t sum1 = vdupq_n_s32(0);
    int32x4_t sum2 = vdupq_n_s32(0);

    // Main loop (note that loop index goes through segments). Unroll 2-wide
    for (int i = 0; i < segments; ++i)
    {
        // Load vector elements to registers
        int16x4_t v11 = vld1_s16(vector1);
        int16x4_t v12 = vld1_s16(vector1 + 4);
        int16x4_t v21 = vld1_s16(vector2);
        int16x4_t v22 = vld1_s16(vector2 + 4);

        sum1 = vmlal_s16(sum1, v11, v21);
        sum2 = vmlal_s16(sum2, v12, v22);

        vector1 += 8;
        vector2 += 8;
    }

    int32x4_t partialSumsNeon = vaddq_s32(sum1, sum2);

    // Store partial sums
    int partialSums[transferSize];
    vst1q_s32(partialSums, partialSumsNeon);

    // Sum up partial sums
    int result = 0;
    for(int i = 0; i < transferSize; i++)
    {
        result += partialSums[i];
    }

    return result;
}

int dotProductNeon4(short* vector1, short* vector2, short len)
{
    const short transferSize = 4;
    short segments = len / (4 * transferSize);

    // 4-element vector of zeros
    int32x4_t sum1 = vdupq_n_s32(0);
    int32x4_t sum2 = vdupq_n_s32(0);
    int32x4_t sum3 = vdupq_n_s32(0);
    int32x4_t sum4 = vdupq_n_s32(0);

    // Main loop (note that loop index goes through segments). Unroll 4-wide
    for (int i = 0; i < segments; ++i)
    {
        // Load vector elements to registers
        int16x4_t v11 = vld1_s16(vector1);
        int16x4_t v12 = vld1_s16(vector1 + 4);
        int16x4_t v13 = vld1_s16(vector1 + 8);
        int16x4_t v14 = vld1_s16(vector1 + 12);
        int16x4_t v21 = vld1_s16(vector2);
        int16x4_t v22 = vld1_s16(vector2 + 4);
        int16x4_t v23 = vld1_s16(vector2 + 8);
        int16x4_t v24 = vld1_s16(vector2 + 12);

        sum1 = vmlal_s16(sum1, v11, v21);
        sum2 = vmlal_s16(sum2, v12, v22);
        sum3 = vmlal_s16(sum3, v13, v23);
        sum4 = vmlal_s16(sum4, v14, v24);

        vector1 += 16;
        vector2 += 16;
    }

    int32x4_t partialSumsNeon = vaddq_s32(sum1, sum2);
    partialSumsNeon = vaddq_s32(partialSumsNeon, sum3);
    partialSumsNeon = vaddq_s32(partialSumsNeon, sum4);

    // Store partial sums
    int partialSums[transferSize];
    vst1q_s32(partialSums, partialSumsNeon);

    // Sum up partial sums
    int result = 0;
    for (int i = 0; i < transferSize; i++)
    {
        result += partialSums[i];
    }

    return result;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_neonintrinsics_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */)
{
    // Ramp length and number of trials
    const int rampLength = 1024;
    const int trials = 1000000;

    // Generate two input vectors
    // (0, 1, ..., rampLength - 1)
    // (100, 101, ..., 100 + rampLength-1)
    auto ramp1 = generateRamp(0, rampLength);
    auto ramp2 = generateRamp(100, rampLength);

    // Without NEON intrinsics
    // Invoke dotProduct and measure performance
    int lastResult = 0;

    Timer timer;
    for (int i = 0; i < trials; i++)
    {
        lastResult = dotProduct(ramp1, ramp2, rampLength);
    }
    auto elapsedMsTime = timer.elapsedMs();

    // With NEON intrinsics
    // Invoke dotProductNeon and measure performance
    int lastResultNeon = 0;
    timer.reset();
    for (int i = 0; i < trials; i++)
    {
        lastResultNeon = dotProductNeon(ramp1, ramp2, rampLength);
    }
    auto elapsedMsTimeNeon = timer.elapsedMs();

    int lastResultNeon2 = 0;
    timer.reset();
    for (int i = 0; i < trials; i++)
    {
        lastResultNeon2 = dotProductNeon2(ramp1, ramp2, rampLength);
    }
    auto elapsedMsTimeNeon2 = timer.elapsedMs();

    int lastResultNeon4 = 0;
    timer.reset();
    for (int i = 0; i < trials; i++)
    {
        lastResultNeon4 = dotProductNeon4(ramp1, ramp2, rampLength);
    }
    auto elapsedMsTimeNeon4 = timer.elapsedMs();

    // Clean up
    delete[] ramp1;
    delete[] ramp2;

    // Display results
    char resultsString[1024];
    snprintf(resultsString, 1024,
        "----==== NO NEON ====----\n\
        Result: %d\
        \nelapsedMs time: %f ms\
        \n\n----==== NEON, no unrolling ====----\n\
        Result: %d\
        \nelapsedMs time: %f ms\
        \n\n----==== NEON 2x unrolling ====----\n\
        Result: %d\
        \nelapsedMs time: %f ms\
        \n\n----==== NEON 4x unrolling ====----\n\
        Result: %d\
        \nelapsedMs time: %f ms",
        lastResult, elapsedMsTime,
        lastResultNeon, elapsedMsTimeNeon,
        lastResultNeon2, elapsedMsTimeNeon2,
        lastResultNeon4, elapsedMsTimeNeon4);

    return env->NewStringUTF(resultsString);
}