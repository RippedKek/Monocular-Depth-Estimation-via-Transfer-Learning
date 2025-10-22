import tensorflow as tf, sys
print("python:", sys.executable)
print("tf.__version__:", getattr(tf, "__version__", None))
print("tf.test.is_built_with_cuda():", getattr(tf.test, "is_built_with_cuda", lambda: "n/a")())
print("tf.config.list_physical_devices('GPU'):", tf.config.list_physical_devices('GPU'))
try:
    print("tf.sysconfig.get_build_info():")
    print(tf.sysconfig.get_build_info())
except Exception as e:
    print("tf.sysconfig.get_build_info() failed:", e)