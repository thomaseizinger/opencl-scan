package at.uastw.hpc.scan;

import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.junit.Before;
import org.junit.Test;

import at.uastw.hpc.scan.opencl.OpenClScan;

/**
 * @author Thomas Eizinger, Senacor Technologies AG.
 */
public class BenchmarkRunner {

    private int[] source;

    private JVMScan jvmScan;
    private OpenClScan openclScan;

    @Before
    public void setUp() throws Exception {
        final URI numbersLocation = BenchmarkRunner.class.getResource("/numbers").toURI();
        this.source = Files.lines(Paths.get(numbersLocation)).mapToInt(Integer::parseInt).toArray();

        jvmScan = new JVMScan();
        openclScan = OpenClScan.create(512, 256);
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
}
