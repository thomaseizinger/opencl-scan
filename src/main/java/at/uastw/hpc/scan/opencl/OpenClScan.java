package at.uastw.hpc.scan.opencl;

import static org.jocl.CL.CL_MEM_COPY_HOST_PTR;
import static org.jocl.CL.CL_MEM_READ_WRITE;

import java.io.File;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Arrays;

import com.github.thomaseizinger.oocl.CLCommandQueue;
import com.github.thomaseizinger.oocl.CLContext;
import com.github.thomaseizinger.oocl.CLDevice;
import com.github.thomaseizinger.oocl.CLKernel;
import com.github.thomaseizinger.oocl.CLMemory;
import com.github.thomaseizinger.oocl.CLPlatform;
import com.github.thomaseizinger.oocl.CLRange;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;

public class OpenClScan {

    private final CLDevice device;

    private final URI kernelURI;
    private static final int LOCAL_SIZE = 8;
    private static final int NUMBER_OF_WORK_GROUPS = 2;
    private static final int NUMBER_OF_WORK_ITEMS = LOCAL_SIZE * NUMBER_OF_WORK_GROUPS;

    private OpenClScan(CLDevice device, URI kernelURI) {
        this.device = device;
        this.kernelURI = kernelURI;
    }

    public static OpenClScan create() {

        final CLPlatform platform = CLPlatform.getFirst().orElseThrow(IllegalStateException::new);
        final CLDevice device = platform.getDevice(CLDevice.DeviceType.GPU).orElseThrow(IllegalStateException::new);

        final URI kernelURI = getKernelURI("/opencl_sum_scan.cl");

        return new OpenClScan(device, kernelURI);
    }

    public int[] sum(int[] source) {

        if (source.length > NUMBER_OF_WORK_ITEMS) {
            throw new IllegalArgumentException("Source array is too long");
        }

        final int nextPowerOf2Length = 32 - Integer.numberOfLeadingZeros(source.length - 1);
        final int desiredLength = (int) Math.pow(2, nextPowerOf2Length);

        final int[] sourceCopy = Arrays.copyOf(source, desiredLength);

        Arrays.fill(sourceCopy, source.length, desiredLength - 1, 0);

        return sumInternal(sourceCopy);
    }

    private int[] sumInternal(int[] source) {

        final int[] result = new int[source.length + 1];
        final int[] workGroupSums = new int[LOCAL_SIZE];

        try (CLContext context = device.createContext()) {
            try (CLKernel scanSum = context.createKernel(new File(kernelURI), "scanSum")) {

                final CLMemory<int[]> inBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, source);
                final CLMemory<int[]> resultBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, result);
                final CLMemory<int[]> workGroupSumsBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, workGroupSums);

                final CLCommandQueue commandQueue = context.createCommandQueue();

                scanSum.setArguments(inBuffer, resultBuffer, workGroupSumsBuffer);
                CL.clSetKernelArg(scanSum.getKernel(), 3, LOCAL_SIZE * 2 * Sizeof.cl_int, new Pointer());

                commandQueue.execute(scanSum, 1, CLRange.of(NUMBER_OF_WORK_ITEMS), CLRange.of(LOCAL_SIZE));

                final CLMemory<int[]> scannedWorkGroupMaxBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, new int[NUMBER_OF_WORK_GROUPS]);

                scanSum.setArguments(workGroupSumsBuffer, scannedWorkGroupMaxBuffer, workGroupSumsBuffer);
                CL.clSetKernelArg(scanSum.getKernel(), 3, NUMBER_OF_WORK_GROUPS * Sizeof.cl_int, new Pointer());

                commandQueue.execute(scanSum, 1, CLRange.of(NUMBER_OF_WORK_GROUPS), CLRange.of(NUMBER_OF_WORK_GROUPS));

                try (CLKernel finalizeScan = context.createKernel(new File(kernelURI), "finalizeScan")) {

                    finalizeScan.setArguments(scannedWorkGroupMaxBuffer, resultBuffer);
                    commandQueue.execute(finalizeScan, 1, CLRange.of(NUMBER_OF_WORK_ITEMS), CLRange.of(NUMBER_OF_WORK_ITEMS / NUMBER_OF_WORK_GROUPS));
                }

                commandQueue.readBuffer(resultBuffer);
                commandQueue.finish();

                return resultBuffer.getData();

            }
        }
    }

    private static URI getKernelURI(String location) {
        try {
            return OpenClScan.class.getResource(location).toURI();
        } catch (URISyntaxException e) {
            throw new IllegalStateException(e);
        }
    }
}
