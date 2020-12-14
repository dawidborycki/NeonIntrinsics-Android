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

int dotProductScalar(short* inputArray1, short* inputArray2, short len)
{
    int result = 0;

    for (short i = 0; i < len; i++)
    {
        result += inputArray1[i] * inputArray2[i];
    }

    return result;
}

int dotProductNeon(short* inputArray1, short* inputArray2, short len)
{
    const int elementsPerIteration = 4;
    int iterations = len / elementsPerIteration;

    // 4-element vector of zeros to accumulate the result
    int32x4_t partialSumsNeon = vdupq_n_s32(0);

    // Main loop
    for (int i = 0; i < iterations; ++i)
    {
        // Load vector elements to registers
        int16x4_t v1 = vld1_s16(inputArray1);
        int16x4_t v2 = vld1_s16(inputArray2);

        partialSumsNeon = vmlal_s16(partialSumsNeon, v1, v2);

        inputArray1 += elementsPerIteration;
        inputArray2 += elementsPerIteration;
    }

	// Armv8 instruction to sum up all the elements into a single scalar
	int result = vaddvq_s32(partialSumsNeon);

	// Calculate the tail
	int tailLength = len % elementsPerIteration;
	while (tailLength--)
	{
		result += *inputArray1 * *inputArray2;
		inputArray1++;
		inputArray2++;
	}

    return result;
}

int dotProductNeon2(short* inputArray1, short* inputArray2, short len)
{
    const int elementsPerIteration = 8;
    int iterations = len / elementsPerIteration;

    // 4-element vectors of zeros to accumulate results within the unrolled loop
    int32x4_t partialSum1 = vdupq_n_s32(0);
    int32x4_t partialSum2 = vdupq_n_s32(0);

    // Main loop, unrolled 2-wide
    for (int i = 0; i < iterations; ++i)
    {
        // Load vector elements to registers
        int16x4_t v11 = vld1_s16(inputArray1);
        int16x4_t v12 = vld1_s16(inputArray1 + 4);
        int16x4_t v21 = vld1_s16(inputArray2);
        int16x4_t v22 = vld1_s16(inputArray2 + 4);

        partialSum1 = vmlal_s16(partialSum1, v11, v21);
        partialSum2 = vmlal_s16(partialSum2, v12, v22);

        inputArray1 += elementsPerIteration;
        inputArray2 += elementsPerIteration;
    }

	// Now sum up the results of the 2 partial sums from the loop
    int32x4_t partialSumsNeon = vaddq_s32(partialSum1, partialSum2);

	// Armv8 instruction to sum up all the elements into a single scalar
	int result = vaddvq_s32(partialSumsNeon);

	// Calculate the tail
	int tailLength = len % elementsPerIteration;
	while (tailLength--)
	{
		result += *inputArray1 * *inputArray2;
		inputArray1++;
		inputArray2++;
	}

    return result;
}

int dotProductNeon4(short* inputArray1, short* inputArray2, short len)
{
    const int elementsPerIteration = 16;
    int iterations = len / elementsPerIteration;

    // 4-element vector of zeros
    int32x4_t partialSum1 = vdupq_n_s32(0);
    int32x4_t partialSum2 = vdupq_n_s32(0);
    int32x4_t partialSum3 = vdupq_n_s32(0);
    int32x4_t partialSum4 = vdupq_n_s32(0);

    // Main loop (note that loop index goes through segments). Unroll 4-wide
    for (int i = 0; i < iterations; ++i)
    {
        // Load vector elements to registers
        int16x4_t v11 = vld1_s16(inputArray1);
        int16x4_t v12 = vld1_s16(inputArray1 + 4);
        int16x4_t v13 = vld1_s16(inputArray1 + 8);
        int16x4_t v14 = vld1_s16(inputArray1 + 12);
        int16x4_t v21 = vld1_s16(inputArray2);
        int16x4_t v22 = vld1_s16(inputArray2 + 4);
        int16x4_t v23 = vld1_s16(inputArray2 + 8);
        int16x4_t v24 = vld1_s16(inputArray2 + 12);

        partialSum1 = vmlal_s16(partialSum1, v11, v21);
        partialSum2 = vmlal_s16(partialSum2, v12, v22);
        partialSum3 = vmlal_s16(partialSum3, v13, v23);
        partialSum4 = vmlal_s16(partialSum4, v14, v24);

        inputArray1 += elementsPerIteration;
        inputArray2 += elementsPerIteration;
    }

	// Now sum up the results of the 4 partial sums from the loop
    int32x4_t partialSumsNeon = vaddq_s32(partialSum1, partialSum2);
    partialSumsNeon = vaddq_s32(partialSumsNeon, partialSum3);
    partialSumsNeon = vaddq_s32(partialSumsNeon, partialSum4);

	// Armv8 instruction to sum up all the elements into a single scalar
	int result = vaddvq_s32(partialSumsNeon);

	// Calculate the tail
	int tailLength = len % elementsPerIteration;
	while (tailLength--)
	{
		result += *inputArray1 * *inputArray2;
		inputArray1++;
		inputArray2++;
	}

    return result;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_neonintrinsics_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */)
{
    // Ramp length and number of trials
    const int rampLength = 1027;
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
        lastResult = dotProductScalar(ramp1, ramp2, rampLength);
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