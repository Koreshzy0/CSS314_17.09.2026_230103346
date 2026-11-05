// Task 22 - OpenMP weak-scaling benchmark: midpoint-rule integration of 4/(1+x^2) on [0,1] (= pi).
// Weak scaling: every thread integrates K steps, so total work = K * P. A fixed serial section of
// SERIAL_STEPS steps (one thread) models Gustafson's non-scaling part (setup / final output).
// Output (stdout, CSV): threads,run,time_s,pi_estimate
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

static double integrate(long first, long last, double h) {          // sum over steps [first,last)
    double sum = 0.0;
    for (long i = first; i < last; ++i) { double x = (i + 0.5) * h; sum += 4.0 / (1.0 + x * x); }
    return sum;
}

int main(int argc, char** argv) {
    int  P      = argc > 1 ? atoi(argv[1]) : 1;                       // threads
    long K      = argc > 2 ? atol(argv[2]) : 50000000L;               // steps per thread (parallel part)
    long SERIAL = argc > 3 ? atol(argv[3]) : 2000000L;                // serial steps (constant)
    int  runs   = argc > 4 ? atoi(argv[4]) : 5;
    int  header = argc > 5 ? atoi(argv[5]) : 0;
    if (header) printf("threads,run,time_s,pi_estimate\n");
    omp_set_num_threads(P);
    long   total = K * (long)P;
    double h     = 1.0 / (double)total;
    for (int r = 0; r < runs; ++r) {
        double t0 = omp_get_wtime();
        // serial section (executed by one thread; identical for every P)
        double s_serial = integrate(0, SERIAL, 1.0 / (double)SERIAL) / (double)SERIAL;   // ~ integral over [0,1] again
        // parallel section: K steps per thread
        double sum = 0.0;
        #pragma omp parallel for schedule(static) reduction(+:sum)
        for (int t = 0; t < P; ++t) sum += integrate((long)t * K, (long)(t + 1) * K, h);
        double pi = sum * h;
        double t1 = omp_get_wtime();
        printf("%d,%d,%.6f,%.10f\n", P, r, t1 - t0, pi + 0.0 * s_serial);
        fflush(stdout);
    }
    return 0;
}
