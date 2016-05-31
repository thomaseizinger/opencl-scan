package at.uastw.hpc.scan;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class SequentialScan {

    public List<Integer> sum(Integer identity, List<Integer> source) {

        final List<Integer> sourceCopy = new ArrayList<>(source);
        final List<Integer> out = new ArrayList<>(Collections.singletonList(identity));

        for (int i = 1; i < sourceCopy.size() + 1; i++) {
            out.add(i, out.get(i - 1) + sourceCopy.get( i - 1 ));
        }

        return out;
    }
}
