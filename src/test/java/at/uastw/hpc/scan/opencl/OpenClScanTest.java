package at.uastw.hpc.scan.opencl;

import org.jocl.CL;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class OpenClScanTest {

    private OpenClScan sut;

    @Before
    public void setUp() throws Exception {
        CL.setExceptionsEnabled(true);
        sut = OpenClScan.create();
    }

    @Test
    public void openClSumScan() throws Exception {

        final int[] source = {1, 2, 3, 4, 5};
        final int[] expected = {0, 1, 3, 6, 10, 15};

        final int[] result = sut.sum(0, source);

        Assert.assertArrayEquals(expected, result);
    }
}