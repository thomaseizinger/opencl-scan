package at.uastw.hpc.scan;

import static org.junit.Assert.*;

import java.util.Arrays;
import java.util.List;

import org.hamcrest.Matchers;
import org.junit.Before;
import org.junit.Test;


public class SequentialScanTest {

    private SequentialScan sut;

    @Before
    public void setUp() throws Exception {
        sut = new SequentialScan();
    }

    @Test
    public void shouldAddElements() throws Exception {

        final List<Integer> out = sut.sum(0, Arrays.asList(1, 2, 3));

        assertThat(out, Matchers.hasItems( 0, 1, 3, 6 ));
    }
}