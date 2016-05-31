package at.uastw.hpc.scan;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.IntFunction;
import java.util.function.Supplier;
import java.util.stream.IntStream;
import java.util.stream.Stream;

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

    }
}