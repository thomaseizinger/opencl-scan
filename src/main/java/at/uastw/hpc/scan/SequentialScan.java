package at.uastw.hpc.scan;

public class SequentialScan {

    public int[] sum(int[] source) {

        final int[] out = new int[source.length];
        out[0] = source[0];

        for (int i = 1; i < source.length; i++) {
            out[i] = out[i - 1] + source[i];
        }

        return out;
    }
}
