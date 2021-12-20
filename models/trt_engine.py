import tensorrt as trt
import numpy as np
import os

import pycuda.driver as cuda
import pycuda.autoinit

# https://blog.csdn.net/hello_dear_you/article/details/120252717
# https://zhuanlan.zhihu.com/p/299845547
# https://stackoverflow.com/questions/59280745/inference-with-tensorrt-engine-file-on-python
# https://blog.csdn.net/hjxu2016/article/details/119796206

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TRTModel:

    def __init__(self, engine_path, max_batch_size=1, input_shape=None) -> None:
        self.engine_path = engine_path
        self.max_batch_size = max_batch_size
        self.input_shape = input_shape
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.inputs, self.outputs, self.bindings, self.stream = \
            self.allocate_buffers(self.input_shape)
        self.context = self.engine.create_execution_context()

    @staticmethod
    def load_engine(trt_runtime, engine_file):
        trt.init_libnvinfer_plugins(None, '')
        with open(engine_file, "rb") as f:
            trt_engine = trt_runtime.deserialize_cuda_engine(f.read())
        return trt_engine

    def allocate_buffers(self, input_shape=None):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            dims = self.engine.get_binding_shape(binding)
            print(dims)
            if dims[-1] == -1: # 输入图片尺寸是动态的情况
                assert(input_shape is not None)
                dims[-2], dims[-1] = input_shape[-2:]
            print(dims)
            size = trt.volume(dims[1:]) * self.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer to device bindings
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream 


    def __call__(self,x:np.ndarray):   
        np.copyto(self.inputs[0].host, x.ravel())

        self.context.set_optimization_profile_async(0,self.stream.handle)
        origin_inputshape = self.context.get_binding_shape(0)
        if(origin_inputshape[0] == -1):
            origin_inputshape[0] = self.max_batch_size
        if(origin_inputshape[-1] == -1):
            origin_inputshape[-2], origin_inputshape[-1] = (x.shape[-2:])

        self.context.set_binding_shape(0, (origin_inputshape))

        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        
        self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
            
        
        self.stream.synchronize()
        return [out.host.reshape(origin_inputshape[0],-1) for out in self.outputs]
        

if __name__=="__main__":
    engine_path = '/workspace/panonet/20211208_qat/outputs/save_engine/engine_quant_False_dynamic_input_True.engine'
    max_batch_size = 128
    shape = (1,3,224,224)
    data = np.random.randint(0,255,(max_batch_size,*shape[1:]))/255
    model = TRTModel(engine_path, max_batch_size, shape)

    out = model(data)
    # print(out)