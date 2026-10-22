import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

public class Task6 {
    static final int THREADS = 4;
    static final int PER_THREAD = 2_000_000;
    static final int EXPECTED = THREADS * PER_THREAD;
    static final int RUNS = 5;

    static final VarHandle AH = MethodHandles.arrayElementVarHandle(int[].class);
    static final int[] plain = new int[1];
    static final AtomicInteger atomicCounter = new AtomicInteger(0);
    static int lockedCounter = 0;
    static final ReentrantLock lock = new ReentrantLock();

    static double timed(Runnable body) throws InterruptedException {
        CountDownLatch go = new CountDownLatch(1);
        Thread[] workers = new Thread[THREADS];
        for (int i = 0; i < THREADS; i++) {
            workers[i] = new Thread(() -> {
                try {
                    go.await();
                } catch (InterruptedException e) {
                    return;
                }
                body.run();
            });
            workers[i].start();
        }
        long start = System.nanoTime();
        go.countDown();
        for (Thread w : workers) w.join();
        return (System.nanoTime() - start) / 1e6;
    }

    static final Runnable UNSYNCHRONIZED = () -> {
        for (int i = 0; i < PER_THREAD; i++) {
            AH.setOpaque(plain, 0, (int) AH.getOpaque(plain, 0) + 1);
        }
    };

    static final Runnable ATOMIC = () -> {
        for (int i = 0; i < PER_THREAD; i++) {
            atomicCounter.incrementAndGet();
        }
    };

    static final Runnable LOCKED = () -> {
        for (int i = 0; i < PER_THREAD; i++) {
            lock.lock();
            try {
                lockedCounter++;
            } finally {
                lock.unlock();
            }
        }
    };

    static double median(double[] values) {
        double[] copy = values.clone();
        Arrays.sort(copy);
        return copy[copy.length / 2];
    }

    public static void main(String[] args) throws InterruptedException {
        System.out.println("Student ID       : 230103346");
        System.out.println("OS / Arch        : " + System.getProperty("os.name") + " "
                + System.getProperty("os.version") + " / " + System.getProperty("os.arch"));
        System.out.println("Java             : " + System.getProperty("java.version"));
        System.out.println("Logical cores    : " + Runtime.getRuntime().availableProcessors());
        System.out.println("Threads          : " + THREADS);
        System.out.printf("Increments/thread: %,d%n", PER_THREAD);
        System.out.printf("Expected final   : %,d%n", EXPECTED);
        System.out.println();

        plain[0] = 0;
        timed(UNSYNCHRONIZED);
        atomicCounter.set(0);
        timed(ATOMIC);
        lockedCounter = 0;
        timed(LOCKED);

        double[] tPlain = new double[RUNS];
        double[] tAtomic = new double[RUNS];
        double[] tLock = new double[RUNS];
        int[] actual = new int[RUNS];

        System.out.println("Unsynchronized runs (no locks, non-atomic read-modify-write):");
        System.out.printf("%4s | %12s | %12s | %9s | %10s%n", "Run", "Actual", "Lost", "Lost (%)", "Time (ms)");
        System.out.println("-".repeat(62));
        for (int r = 0; r < RUNS; r++) {
            plain[0] = 0;
            tPlain[r] = timed(UNSYNCHRONIZED);
            actual[r] = plain[0];
            int lost = EXPECTED - actual[r];
            System.out.printf("%4d | %,12d | %,12d | %8.2f%% | %10.3f%n",
                    r + 1, actual[r], lost, 100.0 * lost / EXPECTED, tPlain[r]);
        }

        for (int r = 0; r < RUNS; r++) {
            atomicCounter.set(0);
            tAtomic[r] = timed(ATOMIC);
            if (atomicCounter.get() != EXPECTED) {
                throw new IllegalStateException("AtomicInteger lost updates");
            }
            lockedCounter = 0;
            tLock[r] = timed(LOCKED);
            if (lockedCounter != EXPECTED) {
                throw new IllegalStateException("Lock version lost updates");
            }
        }

        double meanActual = Arrays.stream(actual).average().orElse(0);
        double meanLostPct = 100.0 * (EXPECTED - meanActual) / EXPECTED;
        double mPlain = median(tPlain);
        double mAtomic = median(tAtomic);
        double mLock = median(tLock);

        System.out.println();
        System.out.println("Summary (unsynchronized, mean of " + RUNS + " runs):");
        System.out.printf("Expected final value : %,d%n", EXPECTED);
        System.out.printf("Actual final value   : %,.0f%n", meanActual);
        System.out.printf("Updates lost         : %.2f%%%n", meanLostPct);
        System.out.println();
        System.out.println("Correct versions (median of " + RUNS + " runs, final value verified = "
                + String.format("%,d", EXPECTED) + "):");
        System.out.printf("%-26s | %10s | %12s | %8s%n", "Version", "Time (ms)", "Penalty (ms)", "Slowdown");
        System.out.println("-".repeat(66));
        System.out.printf("%-26s | %10.3f | %12s | %8s%n", "Unsynchronized (wrong)", mPlain, "-", "1.00x");
        System.out.printf("%-26s | %10.3f | %12.3f | %7.2fx%n", "AtomicInteger", mAtomic, mAtomic - mPlain, mAtomic / mPlain);
        System.out.printf("%-26s | %10.3f | %12.3f | %7.2fx%n", "ReentrantLock", mLock, mLock - mPlain, mLock / mPlain);
    }
}
