# Create an ONNX file with vector add that uses fixed vector sizes.
#
# Code follows from
# https://github.com/onnx/onnx/blob/rel-1.9.0/docs/PythonAPIOverview.md
# and
# https://leimao.github.io/blog/ONNX-Python-API/

import numpy as np
import onnx

n = 10
a = np.ones(n, dtype=np.float32)
b = np.linspace(start=0.0, stop=1.0 * (n - 1), num=n, dtype=np.float32)
c = np.zeros(n, dtype=np.float32)

a_onnx = onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT, [n])
b_onnx = onnx.helper.make_tensor_value_info("b", onnx.TensorProto.FLOAT, [n])
c_onnx = onnx.helper.make_tensor_value_info("c", onnx.TensorProto.FLOAT, [n])


add_node = onnx.helper.make_node(
    name="add1", op_type="Add", inputs=["a", "b"], outputs=["c"]
)

graph_def = onnx.helper.make_graph([add_node], "vector_add", [a_onnx, b_onnx], [c_onnx])

# print(graph_def)
model_def = onnx.helper.make_model(graph_def, producer_name="vector-add-example")

print(model_def)
onnx.checker.check_model(model_def)

onnx.save(model_def, "add_vector.onnx")
