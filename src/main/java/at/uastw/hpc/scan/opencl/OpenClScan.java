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

    public int[] sum(Integer identity, int[] source) {

        final int[] metadata = new int[] { identity, source.length + 1 };
        final int[] result = new int[source.length + 1];

        try (CLContext context = device.createContext()) {
            try (CLKernel scanSum = context.createKernel(new File(kernelURI), "scanSum")) {

                final CLMemory<int[]> inBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, source);
                final CLMemory<int[]> outBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, result);
                final CLMemory<int[]> metBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, metadata);

                scanSum.setArguments(inBuffer, outBuffer, metBuffer);
                CL.clSetKernelArg(scanSum.getKernel(), 3, source.length * Sizeof.cl_int, new Pointer());

                final CLCommandQueue commandQueue = context.createCommandQueue();

                commandQueue.execute(scanSum, 1, CLRange.of(result.length), CLRange.of(result.length));
                commandQueue.readBuffer(outBuffer);
                commandQueue.finish();

                return outBuffer.getData();
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
