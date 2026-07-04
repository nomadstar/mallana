#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

extern void quantize_row_turbo3_0_ref(const float * x, void * y, long long k);
extern void dequantize_row_turbo3_0(const void * x, float * y, long long k);
extern void quantize_row_turbo4_0_ref(const float * x, void * y, long long k);
extern void dequantize_row_turbo4_0(const void * x, float * y, long long k);

static void run_roundtrip_test(const char * name, const float * input, int d) {
    char buf[256];
    float output[128];
    memset(output, 0, sizeof(output));

    printf("Running test: %s\n", name);

    // Test turbo3
    memset(buf, 0, sizeof(buf));
    quantize_row_turbo3_0_ref(input, buf, d);
    dequantize_row_turbo3_0(buf, output, d);
    for (int i = 0; i < d; i++) {
        assert(isfinite(output[i]));
    }

    // Test turbo4
    memset(buf, 0, sizeof(buf));
    quantize_row_turbo4_0_ref(input, buf, d);
    dequantize_row_turbo4_0(buf, output, d);
    for (int i = 0; i < d; i++) {
        assert(isfinite(output[i]));
    }

    printf("  -> OK (finite check passed for turbo3 & turbo4)\n");
}

int main(void) {
    const int d = 128;
    char buf[256];
    float input[128], output[128];
    float mse, cosv, ni, no;

    printf("=== TurboQuant C Round-Trip Test ===\n\n");

    /* Test 1: basis vector */
    memset(input, 0, sizeof(input));
    input[0] = 1.0f;
    quantize_row_turbo3_0_ref(input, buf, d);
    dequantize_row_turbo3_0(buf, output, d);
    printf("Test 1 (turbo3): e0 = [1, 0, ...]\n");
    printf("  In:  [%.6f, %.6f, %.6f, %.6f]\n", (double)input[0], (double)input[1], (double)input[2], (double)input[3]);
    printf("  Out: [%.6f, %.6f, %.6f, %.6f]\n", (double)output[0], (double)output[1], (double)output[2], (double)output[3]);
    mse = cosv = ni = no = 0;
    for (int i = 0; i < d; i++) { mse += (input[i]-output[i])*(input[i]-output[i]); cosv += input[i]*output[i]; ni += input[i]*input[i]; no += output[i]*output[i]; }
    printf("  MSE=%.8f Cosine=%.6f OutNorm=%.6f\n\n", (double)(mse/d), ni > 0 && no > 0 ? (double)(cosv/sqrtf(ni)/sqrtf(no)) : 0.0, (double)sqrtf(no));

    /* Test 2: large-norm vector */
    for (int i = 0; i < d; i++) input[i] = sinf(i*0.1f+0.5f) * 10.0f;
    quantize_row_turbo3_0_ref(input, buf, d);
    dequantize_row_turbo3_0(buf, output, d);
    printf("Test 2 (turbo3): sin*10\n");
    printf("  In:  [%.4f, %.4f, %.4f, %.4f]\n", (double)input[0], (double)input[1], (double)input[2], (double)input[3]);
    printf("  Out: [%.4f, %.4f, %.4f, %.4f]\n", (double)output[0], (double)output[1], (double)output[2], (double)output[3]);
    mse = cosv = ni = no = 0;
    for (int i = 0; i < d; i++) { mse += (input[i]-output[i])*(input[i]-output[i]); cosv += input[i]*output[i]; ni += input[i]*input[i]; no += output[i]*output[i]; }
    printf("  MSE=%.8f Cosine=%.6f InNorm=%.2f OutNorm=%.2f\n\n", (double)(mse/d), (double)(cosv/sqrtf(ni)/sqrtf(no)), (double)sqrtf(ni), (double)sqrtf(no));

    /* Test 3: turbo4 */
    for (int i = 0; i < d; i++) input[i] = cosf(i*0.2f) * 5.0f;
    quantize_row_turbo4_0_ref(input, buf, d);
    dequantize_row_turbo4_0(buf, output, d);
    printf("Test 3 (turbo4): cos*5\n");
    printf("  In:  [%.4f, %.4f, %.4f, %.4f]\n", (double)input[0], (double)input[1], (double)input[2], (double)input[3]);
    printf("  Out: [%.4f, %.4f, %.4f, %.4f]\n", (double)output[0], (double)output[1], (double)output[2], (double)output[3]);
    mse = cosv = ni = no = 0;
    for (int i = 0; i < d; i++) { mse += (input[i]-output[i])*(input[i]-output[i]); cosv += input[i]*output[i]; ni += input[i]*input[i]; no += output[i]*output[i]; }
    printf("  MSE=%.8f Cosine=%.6f\n\n", (double)(mse/d), (double)(cosv/sqrtf(ni)/sqrtf(no)));

    printf("=== Extreme-Input Test Cases (Robustness) ===\n\n");

    // 1. Zero Vector: All inputs set to 0.0f
    float input_zero[128];
    for (int i = 0; i < d; i++) input_zero[i] = 0.0f;
    run_roundtrip_test("Zero Vector", input_zero, d);

    // 2. NaN Input: Vector containing NAN values
    float input_nan[128];
    for (int i = 0; i < d; i++) {
        input_nan[i] = (i % 10 == 0) ? NAN : (float)i;
    }
    run_roundtrip_test("NaN Input", input_nan, d);

    // 3. Inf Input: Vector containing INFINITY and -INFINITY values
    float input_inf[128];
    for (int i = 0; i < d; i++) {
        if (i % 10 == 0) {
            input_inf[i] = INFINITY;
        } else if (i % 10 == 5) {
            input_inf[i] = -INFINITY;
        } else {
            input_inf[i] = (float)i;
        }
    }
    run_roundtrip_test("Inf Input", input_inf, d);

    // 4. Single Mass Outlier: Standard input vector with a single extremely large outlier (1e6)
    float input_outlier[128];
    for (int i = 0; i < d; i++) {
        input_outlier[i] = sinf(i * 0.1f);
    }
    input_outlier[42] = 1e6f;
    run_roundtrip_test("Single Mass Outlier (1e6)", input_outlier, d);

    // 5. Laplace Distribution with Outliers: Laplace generated values interspersed with 1e5 outliers
    float input_laplace[128];
    srand(1337);
    for (int i = 0; i < d; i++) {
        if (i % 15 == 0) {
            input_laplace[i] = 1e5f;
        } else if (i % 15 == 7) {
            input_laplace[i] = -1e5f;
        } else {
            float u = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f) - 0.5f;
            float sgn = (u < 0.0f) ? -1.0f : 1.0f;
            input_laplace[i] = -2.0f * sgn * logf(1.0f - 2.0f * fabsf(u));
        }
    }
    run_roundtrip_test("Laplace Distribution with Outliers", input_laplace, d);

    // 6. Combined Extreme Cases: Mixed vector of normal numbers, NaNs, Infs, and outliers
    float input_combined[128];
    for (int i = 0; i < d; i++) {
        if (i % 12 == 0) {
            input_combined[i] = NAN;
        } else if (i % 12 == 3) {
            input_combined[i] = INFINITY;
        } else if (i % 12 == 6) {
            input_combined[i] = -INFINITY;
        } else if (i % 12 == 9) {
            input_combined[i] = 1e6f;
        } else {
            input_combined[i] = cosf(i * 0.2f) * 2.0f;
        }
    }
    run_roundtrip_test("Combined Extreme Cases", input_combined, d);

    printf("\n=== Done ===\n");
    return 0;
}
