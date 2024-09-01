# Run the vector add ONNX file created with create_vector_add.py
# Code follows
# https://onnxruntime.ai/docs/api/python/tutorial.html
import numpy as np
import onnxruntime as rt

n = 10
a = np.ones(n, dtype=np.float32)
b = np.linspace(start=0.0, stop=1.0 * (n - 1), num=n, dtype=np.float32)

print(rt.get_available_providers())
# sess = rt.InferenceSession("add_vector.onnx", providers=rt.get_available_providers())
sess = rt.InferenceSession("add_vector.onnx", providers=["CPUExecutionProvider"])

input_names = [si.name for si in sess.get_inputs()]
print("input name", input_names)

c = sess.run(None, {"a": a, "b": b})
print(c)
