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

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;

public class OpenClScan {

    private static final Logger LOGGER = LogManager.getLogger(OpenClScan.class);

    private final CLDevice device;

    private final URI kernelURI;
    private final int localSize;
    private final int numberOfWorkItems;
    private int numberOfWorkGroups;

    private OpenClScan(CLDevice device, URI kernelURI, int localSize, int numberOfWorkGroups) {
        this.device = device;
        this.kernelURI = kernelURI;
        this.localSize = localSize;
        this.numberOfWorkGroups = numberOfWorkGroups;
        this.numberOfWorkItems = this.localSize * this.numberOfWorkGroups;
    }

    public static OpenClScan create(int localSize, int numberOfWorkGroups) {

        final CLPlatform platform = CLPlatform.getFirst().orElseThrow(IllegalStateException::new);
        final CLDevice device = platform.getDevice(CLDevice.DeviceType.GPU).orElseThrow(IllegalStateException::new);

        final URI kernelURI = getKernelURI("/opencl_sum_scan.cl");

        return new OpenClScan(device, kernelURI, localSize, numberOfWorkGroups);
    }

    public int[] sum(int[] source) {

        if (source.length > numberOfWorkItems * 2) {
            throw new IllegalArgumentException("Source array is too long");
        }

        final int nextPowerOf2Length = 32 - Integer.numberOfLeadingZeros(source.length - 1);
        final int desiredLength = (int) Math.pow(2, nextPowerOf2Length);

        if (desiredLength > source.length) {
            final int[] sourceCopy = Arrays.copyOf(source, desiredLength);
            Arrays.fill(sourceCopy, source.length, desiredLength - 1, 0);

            return sumInternal(sourceCopy);
        } else {
            return sumInternal(source);
        }
    }

    private int[] sumInternal(int[] source) {

        final int[] result = new int[source.length];
        final int[] workGroupSums = new int[localSize];

        final long start = System.nanoTime();

        try (CLContext context = device.createContext()) {
            try (CLKernel scanSum = context.createKernel(new File(kernelURI), "scanSum")) {

                final CLMemory<int[]> inBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, source);
                final CLMemory<int[]> resultBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                        result);
                final CLMemory<int[]> workGroupSumsBuffer = context.createBuffer(CL_MEM_READ_WRITE |
                        CL_MEM_COPY_HOST_PTR, workGroupSums);

                final CLCommandQueue commandQueue = context.createCommandQueue();

                scanSum.setArguments(inBuffer, resultBuffer, workGroupSumsBuffer);
                CL.clSetKernelArg(scanSum.getKernel(), 3, localSize * 2 * Sizeof.cl_int, new Pointer());

                commandQueue.execute(scanSum, 1, CLRange.of(numberOfWorkItems), CLRange.of(localSize));

                final CLMemory<int[]> scannedWorkGroupMaxBuffer = context.createBuffer(CL_MEM_READ_WRITE |
                        CL_MEM_COPY_HOST_PTR, new int[numberOfWorkGroups]);

                scanSum.setArguments(workGroupSumsBuffer, scannedWorkGroupMaxBuffer, workGroupSumsBuffer);
                CL.clSetKernelArg(scanSum.getKernel(), 3, numberOfWorkGroups * Sizeof.cl_int, new Pointer());

                commandQueue.execute(scanSum, 1, CLRange.of(numberOfWorkGroups / 2), CLRange.of(numberOfWorkGroups / 2));

                try (CLKernel finalizeScan = context.createKernel(new File(kernelURI), "finalizeScan")) {

                    finalizeScan.setArguments(scannedWorkGroupMaxBuffer, resultBuffer);
                    commandQueue.execute(finalizeScan, 1, CLRange.of(numberOfWorkItems), CLRange.of(localSize));
                }

                commandQueue.readBuffer(resultBuffer);
                commandQueue.finish();

                return resultBuffer.getData();

            }
        } finally {
            LOGGER.info("Execution time: {} µs", (System.nanoTime() - start) / 1000);
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
