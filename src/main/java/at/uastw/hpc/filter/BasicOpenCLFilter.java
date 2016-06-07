package at.uastw.hpc.filter;

import static org.jocl.CL.CL_MEM_COPY_HOST_PTR;
import static org.jocl.CL.CL_MEM_READ_ONLY;
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

import at.uastw.hpc.scan.OpenClScan;

/**
 * @author Thomas Eizinger, Senacor Technologies AG.
 */
public class BasicOpenCLFilter {

    private static final Logger LOGGER = LogManager.getLogger(BasicOpenCLFilter.class);

    private final CLDevice device;

    private final OpenClScan scan;
    private final URI kernelURI;
    private final int localSize;
    private final int numberOfWorkItems;

    private BasicOpenCLFilter(CLDevice device, OpenClScan scan, URI kernelURI, int localSize, int numberOfWorkGroups) {
        this.device = device;
        this.scan = scan;
        this.kernelURI = kernelURI;
        this.localSize = localSize;
        this.numberOfWorkItems = this.localSize * numberOfWorkGroups;
    }

    public static BasicOpenCLFilter create(int localSize, int numberOfWorkGroups) {

        final CLPlatform platform = CLPlatform.getFirst().orElseThrow(IllegalStateException::new);
        final CLDevice device = platform.getDevice(CLDevice.DeviceType.GPU).orElseThrow(IllegalStateException::new);

        final URI kernelURI = getKernelURI("/opencl_basic_filter.cl");

        final OpenClScan scan = OpenClScan.create(localSize, numberOfWorkGroups);

        return new BasicOpenCLFilter(device, scan, kernelURI, localSize, numberOfWorkGroups);
    }

    public int[] filterGreaterThan(int[] source, int threshold) {
        return filterInternal(source, threshold, "filterGreaterThan");
    }

    public int[] filterLessThan(int[] source, int threshold) {
        return filterInternal(source, threshold, "filterLessThan");
    }

    public int[] filterEquals(int[] source, int candidate) {
        return filterInternal(source, candidate, "filterEquals");
    }

    private int[] filterInternal(int[] source, int threshold, String kernelName) {
        int[] addresses = new int[source.length];
        int[] metadata = new int[] { threshold };

        final long start = System.nanoTime();

        try (CLContext context = device.createContext()) {
            try (CLKernel filterGreaterThan = context.createKernel(new File(kernelURI), kernelName)) {
                try(final CLMemory<int[]> inBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, source);
                    final CLMemory<int[]> addressBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, addresses);
                    final CLMemory<int[]> metadataBuffer = context.createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, metadata)) {

                    final CLCommandQueue commandQueue = context.createCommandQueue();

                    filterGreaterThan.setArguments(inBuffer, addressBuffer, metadataBuffer);

                    commandQueue.execute(filterGreaterThan, 1, CLRange.of(numberOfWorkItems), CLRange.of(localSize));

                    commandQueue.readBuffer(addressBuffer);

                    final int[] scannedAddresses = scan.sum(addressBuffer.getData());

                    LOGGER.info("Scanned addresses {}", Arrays.toString(scannedAddresses));

                    int[] output = new int[source.length];

                    try (final CLKernel applyFilter = context.createKernel(new File(kernelURI), "applyFilter");
                         final CLMemory<int[]> outBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, output);
                         final CLMemory<int[]> scannedAddressBuffer = context.createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, scannedAddresses)) {

                        applyFilter.setArguments(inBuffer, outBuffer, scannedAddressBuffer);

                        commandQueue.execute(applyFilter, 1, CLRange.of(numberOfWorkItems), CLRange.of(localSize));
                        commandQueue.readBuffer(outBuffer);
                        commandQueue.finish();

                        return outBuffer.getData();
                    }
                }
            }
        } finally {
            LOGGER.info("Execution time: {} µs", (System.nanoTime() - start) / 1000);
        }
    }

    private static URI getKernelURI(String location) {
        try {
            return BasicOpenCLFilter.class.getResource(location).toURI();
        } catch (URISyntaxException e) {
            throw new IllegalStateException(e);
        }
    }

}
