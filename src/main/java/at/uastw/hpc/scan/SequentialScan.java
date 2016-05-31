package at.uastw.hpc.scan;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class SequentialScan {

    public int[] sum(Integer identity, int[] source) {

        final List<Integer> out = new ArrayList<>(Collections.singletonList(identity));

        for (int i = 1; i < source.length + 1; i++) {
            out.add(i, out.get(i - 1) + source[i - 1]);
        }

        return out.stream().mapToInt(i -> i).toArray();
    }
}
