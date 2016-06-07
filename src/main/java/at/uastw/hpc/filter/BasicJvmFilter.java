package at.uastw.hpc.filter;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * @author Thomas Eizinger, Senacor Technologies AG.
 */
public class BasicJvmFilter {

    private static final Logger LOGGER = LogManager.getLogger(BasicJvmFilter.class);

    public int[] filterGreaterThan(int[] source, int threshold) {

        int[] output = new int[source.length];
        int currentIndex = 1;

        final long start = System.nanoTime();

        for (int el : source) {
            if (el > threshold) {
                output[currentIndex++] = el;
            }
        }

        LOGGER.info("Execution time: {} µs", (System.nanoTime() - start) / 1000);

        return output;
    }
}
