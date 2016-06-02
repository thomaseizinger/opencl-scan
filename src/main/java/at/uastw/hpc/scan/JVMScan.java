package at.uastw.hpc.scan;

public class JVMScan {

    public int[] sum(int[] source) {

        final long start = System.nanoTime();

        final int[] out = new int[source.length];
        out[0] = source[0];

        for (int i = 1; i < source.length; i++) {
            out[i] = out[i - 1] + source[i];
        }

        final long stop = System.nanoTime();

        System.out.println(String.format("Execution time: %d µs", (stop - start) / 1000));

        return out;
    }
}
