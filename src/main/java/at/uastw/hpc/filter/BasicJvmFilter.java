package at.uastw.hpc.filter;

import java.util.function.Predicate;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * @author Thomas Eizinger, Senacor Technologies AG.
 */
public class BasicJvmFilter {

    private static final Logger LOGGER = LogManager.getLogger(BasicJvmFilter.class);

    public int[] filterGreaterThan(int[] source, int threshold) {
        return filterInternal(source, (i) -> i > threshold);
    }

    public int[] filterLessThan(int[] source, int threshold) {
        return filterInternal(source, (i) -> i < threshold);
    }

    public int[] filterEquals(int[] source, int threshold) {
        return filterInternal(source, (i) -> i == threshold);
    }

    private int[] filterInternal(int[] source, Predicate<Integer> predicate) {
        int[] output = new int[source.length];
        int currentIndex = 1;

        final long start = System.nanoTime();

        for (int el : source) {
            if (predicate.test(el)) {
                output[currentIndex++] = el;
            }
        }

        LOGGER.info("Execution time: {} µs", (System.nanoTime() - start) / 1000);

        return output;
    }
}
