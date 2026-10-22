import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.Arrays;

public class Task3 {
    static final long ITERATIONS = 50_000_000L;
    static final int RUNS = 5;
    static final int STRIDE = 16;
    static final VarHandle AH = MethodHandles.arrayElementVarHandle(long[].class);

    enum Mode { UNPADDED, PADDED, LOCAL }

        static double run(Mode mode, int threads, long iterations) throws InterruptedException {
        final int stride = (mode == Mode.PADDED) ? STRIDE : 1;
        final long[] counts = new long[threads * stride];
        Thread[] workers = new Thread[threads];

        for (int t = 0; t < threads; t++) {
            final int idx = t * stride;
            workers[t] = switch (mode) {
                case UNPADDED, PADDED -> new Thread(() -> {
                    for (long k = 0; k < iterations; k++) {
                        AH.setOpaque(counts, idx, (long) AH.getOpaque(counts, idx) + 1L);
                    }
                });
                case LOCAL -> new Thread(() -> {
                    long local = 0;
                    for (long k = 0; k < iterations; k++) {
                        local++;
                    }
                    counts[idx] = local;
                });
            };
        }

        long start = System.nanoTime();
        for (Thread w : workers) w.start();
        for (Thread w : workers) w.join();
        double ms = (System.nanoTime() - start) / 1e6;

        for (int t = 0; t < threads; t++) {
            if (counts[t * stride] != iterations) {
                throw new IllegalStateException("Wrong count in " + mode + " for thread " + t);
            }
        }
        return ms;
    }

    static double median(Mode mode, int threads) throws InterruptedException {
        run(mode, threads, ITERATIONS / 10);
        double[] times = new double[RUNS];
        for (int i = 0; i < RUNS; i++) times[i] = run(mode, threads, ITERATIONS);
        Arrays.sort(times);
        return times[RUNS / 2];
    }

    public static void main(String[] args) throws InterruptedException {
        int threads = (args.length > 0) ? Integer.parseInt(args[0])
                                        : Runtime.getRuntime().availableProcessors();

        System.out.println("Student ID       : 230103346");
        System.out.println("OS / Arch        : " + System.getProperty("os.name") + " "
                + System.getProperty("os.version") + " / " + System.getProperty("os.arch"));
        System.out.println("Java             : " + System.getProperty("java.version"));
        System.out.println("Threads (N)      : " + threads);
        System.out.println("Increments/thread: " + String.format("%,d", ITERATIONS));
        System.out.println("Method           : median of " + RUNS + " runs, padding stride = "
                + STRIDE + " longs (" + STRIDE * 8 + " bytes)");
        System.out.println();

        double unpadded = median(Mode.UNPADDED, threads);
        double padded   = median(Mode.PADDED, threads);
        double local    = median(Mode.LOCAL, threads);

        System.out.printf("Unpadded Shared  : %10.3f ms%n", unpadded);
        System.out.printf("Padded (128 B)   : %10.3f ms%n", padded);
        System.out.printf("Local variable   : %10.3f ms%n", local);
        System.out.println();
        System.out.printf("Ratio Unpadded / Padded : %6.2fx%n", unpadded / padded);
        System.out.printf("Ratio Unpadded / Local  : %6.2fx%n", unpadded / local);
        System.out.println("All counters verified: each equals " + String.format("%,d", ITERATIONS));
    }
}
