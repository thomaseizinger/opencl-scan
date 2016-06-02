package at.uastw.hpc.scan.opencl;

import static junitparams.JUnitParamsRunner.$;

import java.util.Arrays;
import java.util.stream.IntStream;

import org.jocl.CL;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import at.uastw.hpc.scan.JVMScan;
import junitparams.JUnitParamsRunner;
import junitparams.Parameters;

@RunWith(JUnitParamsRunner.class)
public class OpenClScanTest {

    private OpenClScan sut;
    private JVMScan jvmScan;

    @Before
    public void setUp() throws Exception {
        CL.setExceptionsEnabled(true);
        jvmScan = new JVMScan();
        sut = OpenClScan.create();
    }

    @Test
    @Parameters
    public void testSum(String reason, int[] source) throws Exception {

        final int[] result = sut.sum(source);
        final int[] expected = jvmScan.sum(source);

        final int[] trimmedResult = Arrays.stream(result).limit(source.length).toArray();

        Assert.assertArrayEquals(reason, expected, trimmedResult);
    }

    public Object[] parametersForTestSum() {
        return $(
                $("should sum small array", range(1, 64)),
                $("should sum array with non base2 length", range(1, 13)),
                $("should sum big array", range(1, 130000))
        );
    }

    private static int[] range(int from, int to) {
        return IntStream.range(from, to).toArray();
    }
}