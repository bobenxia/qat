
import onnxruntime

class ONNXModel():
    def __init__(self, onnx_path) -> None:
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        
    def get_input_name(self, onnx_session):
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_output_name(self, onnx_session):
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_feed(self, input_name, image_numpy):
        """input_feed={self.input_name: image_numpy}"""
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def forward(self, image_numpy):
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        scores = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return scores
    