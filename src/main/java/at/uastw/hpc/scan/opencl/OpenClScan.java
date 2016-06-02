package at.uastw.hpc.scan.opencl;

import static org.jocl.CL.CL_MEM_COPY_HOST_PTR;
import static org.jocl.CL.CL_MEM_READ_ONLY;
import static org.jocl.CL.CL_MEM_READ_WRITE;

import java.awt.image.BufferedImage;
import java.io.File;
import java.net.URI;
import java.net.URISyntaxException;

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
import org.jocl.cl_mem;

public class OpenClScan {

    private final CLDevice device;

    private final URI kernelURI;

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

        int localSize = 512;
        int numberOfWorkGroups = 256;

        int numberOfWorkItems = localSize * numberOfWorkGroups;

        final int[] result = new int[source.length + 1];
        final int[] workGroupSums = new int[ localSize ];

        try (CLContext context = device.createContext()) {
            try (CLKernel scanSum = context.createKernel(new File(kernelURI), "scanSum")) {

                final CLMemory<int[]> inBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, source);
                final CLMemory<int[]> resultBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, result);
                final CLMemory<int[]> workGroupSumsBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, workGroupSums);

                final CLCommandQueue commandQueue = context.createCommandQueue();

                scanSum.setArguments(inBuffer, resultBuffer, workGroupSumsBuffer);
                CL.clSetKernelArg(scanSum.getKernel(), 3, localSize * Sizeof.cl_int, new Pointer());

                commandQueue.execute(scanSum, 1, CLRange.of(numberOfWorkItems), CLRange.of(localSize));

                final CLMemory<int[]> scannedWorkGroupMaxBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, new int[numberOfWorkGroups]);

                scanSum.setArguments(workGroupSumsBuffer, scannedWorkGroupMaxBuffer, workGroupSumsBuffer);
                CL.clSetKernelArg(scanSum.getKernel(), 3, numberOfWorkGroups * Sizeof.cl_int, new Pointer());

                commandQueue.execute(scanSum, 1, CLRange.of(numberOfWorkGroups + 1), CLRange.of(numberOfWorkGroups + 1));

                try (CLKernel finalizeScan = context.createKernel(new File(kernelURI), "finalizeScan")) {

                    finalizeScan.setArguments(scannedWorkGroupMaxBuffer, resultBuffer);
                    commandQueue.execute(finalizeScan, 1, CLRange.of(source.length + 1), CLRange.of(numberOfWorkItems / numberOfWorkGroups));
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
