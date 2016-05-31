package at.uastw.hpc.scan.opencl;

import java.util.stream.IntStream;

import org.jocl.CL;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import at.uastw.hpc.scan.SequentialScan;

public class OpenClScanTest {

    private OpenClScan sut;

    @Before
    public void setUp() throws Exception {
        CL.setExceptionsEnabled(true);
        sut = OpenClScan.create();
    }

    @Test
    public void openClSumScan() throws Exception {

        final int[] source = IntStream.range(1, 4).toArray();
        final int[] expected = new SequentialScan().sum(0, source);

        final int[] result = sut.sum(0, source);

        Assert.assertArrayEquals(expected, result);
    }
}