import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.input_binding = self.engine.get_binding_index("input")
        self.output_binding = self.engine.get_binding_index("logits")

        self.input_shape = self.engine.get_binding_shape(self.input_binding)
        self.output_shape = self.engine.get_binding_shape(self.output_binding)

        self.input_dtype = trt.nptype(self.engine.get_binding_dtype(self.input_binding))
        self.output_dtype = trt.nptype(self.engine.get_binding_dtype(self.output_binding))

        self.d_input = None
        self.d_output = None

    def allocate_buffers(self, batch_size):
        input_shape = (batch_size, 3, 3, 224, 224)
        output_shape = (batch_size, 10)

        self.context.set_binding_shape(self.input_binding, input_shape)

        input_size = np.prod(input_shape) * np.dtype(self.input_dtype).itemsize
        output_size = np.prod(output_shape) * np.dtype(self.output_dtype).itemsize

        self.d_input = cuda.mem_alloc(input_size)
        self.d_output = cuda.mem_alloc(output_size)

        self.bindings = [int(self.d_input), int(self.d_output)]

    def infer(self, x: np.ndarray):
        """
        x: np.ndarray (B, 3, 3, 224, 224), float32 or float16
        """
        batch_size = x.shape[0]

        if self.d_input is None:
            self.allocate_buffers(batch_size)

        cuda.memcpy_htod_async(self.d_input, x, self.stream)

        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        output = np.empty((batch_size, 10), dtype=self.output_dtype)
        cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
        self.stream.synchronize()

        return output
