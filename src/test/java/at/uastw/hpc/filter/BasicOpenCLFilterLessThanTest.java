package at.uastw.hpc.filter;

import static junitparams.JUnitParamsRunner.$;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

import org.jocl.CL;
import org.junit.Assert;
import org.junit.Before;
import org.junit.FixMethodOrder;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.MethodSorters;

import junitparams.JUnitParamsRunner;
import junitparams.Parameters;

/**
 * @author Thomas Eizinger, Senacor Technologies AG.
 */
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
@RunWith(JUnitParamsRunner.class)
public class BasicOpenCLFilterLessThanTest {

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
    public void shouldFilterLessThan(int threshold, int[] source) throws Exception {

        final int[] prependedSource = new int[source.length + 1];
        System.arraycopy(source, 0, prependedSource, 1, source.length);

        final int[] result = sut.filterLessThan(prependedSource, threshold);
        final int[] expected = basicJvmFilter.filterLessThan(prependedSource, threshold);

        final int[] trimmedResult = Arrays.stream(result).skip(1).toArray();
        final int[] trimmedExpected = Arrays.stream(expected).skip(1).toArray();

        Assert.assertArrayEquals(trimmedExpected, trimmedResult);
    }

    public Object[] parametersForShouldFilterLessThan() {
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