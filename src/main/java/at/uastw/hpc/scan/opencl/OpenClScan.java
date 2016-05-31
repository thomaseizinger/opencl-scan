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

        final int[] metadata = new int[] { identity, source.length };
        final int[] result = new int[source.length];

        try (CLContext context = device.createContext()) {
            try (CLKernel scanSum = context.createKernel(new File(kernelURI), "scanSum")) {

                final cl_mem mem_Metadata = CL.clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int * metadata.length, Pointer.to(metadata), null);
                final cl_mem mem_Source = CL.clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int * source.length, Pointer.to(source), null);
                final cl_mem mem_Result = CL.clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int * result.length, Pointer.to(result), null);

                CL.clSetKernelArg(scanSum.getKernel(), 0, Sizeof.cl_mem, Pointer.to(mem_Metadata));
                CL.clSetKernelArg(scanSum.getKernel(), 1, Sizeof.cl_mem, Pointer.to(mem_Source));
                CL.clSetKernelArg(scanSum.getKernel(), 2, Sizeof.cl_mem, Pointer.to(mem_Result));
                CL.clSetKernelArg(scanSum.getKernel(), 3, Sizeof.cl_mem, new Pointer());

                final CLCommandQueue commandQueue = context.createCommandQueue();

                commandQueue.execute(scanSum, 1, CLRange.of(source.length), CLRange.of(1, 1));
                commandQueue.finish();

                return new int[] { };
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
