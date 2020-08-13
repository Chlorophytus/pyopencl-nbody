import numpy as np
import pkg_resources
import pyoclnbody.log
import pyopencl as cl
import pyopencl.cltypes


class ParticleSystem(pyoclnbody.log.Loggable):
    """An OpenCL-accelerated N-body particle system"""

    # How many particles in the system?
    NUM_PARTICLES: int = 64

    # Memory flags
    mf = cl.mem_flags

    def __init__(self):
        super().__init__()
        # Create a context
        self.log(pyoclnbody.log.logging.INFO, "OpenCL Context creation")
        self.ctx = cl.create_some_context()
        self.log(pyoclnbody.log.logging.INFO, "OpenCL Context success")

        # Create a command queue
        self.log(pyoclnbody.log.logging.INFO, "OpenCL CommandQueue creation")
        self.queue = cl.CommandQueue(self.ctx)
        self.log(pyoclnbody.log.logging.INFO, "OpenCL CommandQueue success")

        # Load a kernel
        self.log(pyoclnbody.log.logging.INFO, "OpenCL Kernel creation")
        name = pkg_resources.resource_filename(
            'pyoclnbody', "resources/kParticleSystem.ocl")
        with open(name, 'r') as f:
            self.program = cl.Program(self.ctx, ''.join(f.readlines())).build(
                options=['-cl-std=CL1.2', '-w'])
        self.log(pyoclnbody.log.logging.INFO, "OpenCL Kernel success")

        # Create the variables
        self.r_data = np.array([cl.cltypes.make_float4(np.random.Generator(np.random.MT19937()).normal(), np.random.Generator(np.random.MT19937(
            )).normal(), np.random.Generator(np.random.MT19937()).normal(), np.random.Generator(np.random.MT19937()).normal()) for _ in range(128)])

        self.w_data = np.empty_like(self.r_data)
        self._pending_data = False

    def set_data(self):
        # Read off the data
        self.log(pyoclnbody.log.logging.INFO, "OpenCL Buffer read pending")
        r_buff = cl.Buffer(self.ctx, self.mf.READ_ONLY |
                           self.mf.COPY_HOST_PTR, hostbuf=self.r_data)
        self.log(pyoclnbody.log.logging.INFO, "OpenCL Buffer read complete")

        # Write-only memory for the OpenCL device to generate into
        w_buff = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, self.r_data.nbytes)

        self.program.calc_forces(
            self.queue, self.r_data.shape, None, r_buff, w_buff)

        # Copy the written data result from the buffer
        self.log(pyoclnbody.log.logging.INFO, "OpenCL Buffer copy pending")
        cl.enqueue_copy(self.queue, self.w_data, w_buff)
        self.log(pyoclnbody.log.logging.INFO, "OpenCL Buffer copy complete")

        # The data is now pending
        self._pending_data = True

    def get_data(self) -> np.array:
        self.r_data = self.w_data
        self._pending_data = False
        return self.w_data

    def pending_data(self) -> bool:
        return self._pending_data
