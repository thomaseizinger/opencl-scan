package at.uastw.hpc.filter;

import static junitparams.JUnitParamsRunner.$;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

import org.jocl.CL;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import at.uastw.hpc.filter.BasicJvmFilter;
import at.uastw.hpc.filter.BasicOpenCLFilter;
import junitparams.JUnitParamsRunner;
import junitparams.Parameters;

/**
 * @author Thomas Eizinger, Senacor Technologies AG.
 */
@RunWith(JUnitParamsRunner.class)
public class BasicOpenCLFilterTest {

    private BasicOpenCLFilter sut;
    private BasicJvmFilter basicJvmFilter;

    @Before
    public void setUp() throws Exception {

        CL.setExceptionsEnabled(true);

        sut = BasicOpenCLFilter.create(512, 2);
        basicJvmFilter = new BasicJvmFilter();
    }

    @Test
    @Parameters
    public void shouldFilterElements(int threshold, int[] source) throws Exception {

        final int[] prependedSource = new int[source.length + 1];
        System.arraycopy(source, 0, prependedSource, 1, source.length);

        final int[] result = sut.filterGreaterThan(prependedSource, threshold);
        final int[] expected = basicJvmFilter.filterGreaterThan(prependedSource, threshold);

        final int[] trimmedResult = Arrays.stream(result).skip(1).toArray();
        final int[] trimmedExpected = Arrays.stream(expected).skip(1).toArray();

        Assert.assertArrayEquals(trimmedExpected, trimmedResult);
    }

    public Object[] parametersForShouldFilterElements() {
        final Object[] testCases = $(
                $(200, new int[] {400, 201, 300, 5}),
                $(200, randomElements(300))
        );
        return Arrays.stream(testCases).toArray();
    }

    private static int[] randomElements(int numberOfElements) {

        final Random rnd = new Random();

        return IntStream.generate( () -> rnd.nextInt(500) ).limit(numberOfElements).toArray();
    }
}