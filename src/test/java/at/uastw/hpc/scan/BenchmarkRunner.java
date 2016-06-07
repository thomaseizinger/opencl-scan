package at.uastw.hpc.scan;

import java.util.Random;
import java.util.stream.IntStream;

import org.junit.Before;
import org.junit.Test;

import at.uastw.hpc.filter.BasicJvmFilter;
import at.uastw.hpc.filter.BasicOpenCLFilter;

/**
 * @author Thomas Eizinger, Senacor Technologies AG.
 */
public class BenchmarkRunner {

    private int[] source;

    private JVMScan jvmScan;
    private OpenClScan openclScan;
    private BasicJvmFilter basicJvmFilter;
    private BasicOpenCLFilter basicOpenCLFilter;

    @Before
    public void setUp() throws Exception {
        this.source = randomElements(512 * 2);

        jvmScan = new JVMScan();
        openclScan = OpenClScan.create(512, 2);

        basicJvmFilter = new BasicJvmFilter();
        basicOpenCLFilter = BasicOpenCLFilter.create(512, 2);
    }

    @Test
    public void benchmarkJVMScan() throws Exception {
        for (int i = 0; i < 10; i++) {
            jvmScan.sum(source);
        }
    }

    @Test
    public void benchmarkOpenCLScan() throws Exception {
        for (int i = 0; i < 10; i++) {
            openclScan.sum(source);
        }
    }

    @Test
    public void benchmarkJVMFilter() throws Exception {
        for (int i = 0; i < 10; i++) {
            basicJvmFilter.filterGreaterThan(source, 812);
        }
    }

    @Test
    public void benchmarkOpenCLFilter() throws Exception {
        for (int i = 0; i < 10; i++) {
            basicOpenCLFilter.filterGreaterThan(source, 812);
        }
    }

    private static int[] randomElements(int numberOfElements) {

        final Random rnd = new Random();

        return IntStream.generate( () -> rnd.nextInt(1000) ).limit(numberOfElements).toArray();
    }
}
