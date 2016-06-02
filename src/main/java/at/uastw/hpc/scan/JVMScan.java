package at.uastw.hpc.scan;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class JVMScan {

    private static final Logger LOGGER = LogManager.getLogger(JVMScan.class);

    public int[] sum(int[] source) {

        final long start = System.nanoTime();

        final int[] out = new int[source.length];
        out[0] = source[0];

        for (int i = 1; i < source.length; i++) {
            out[i] = out[i - 1] + source[i];
        }

        LOGGER.info("Execution time: {} µs", (System.nanoTime() - start) / 1000);

        return out;
    }
}
