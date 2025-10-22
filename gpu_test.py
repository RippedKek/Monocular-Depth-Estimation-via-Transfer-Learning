"""
gpu_test.py

Quick GPU smoke-test script for TensorFlow and PyTorch.
- Prints detected GPUs
- Runs a small matrix multiply on GPU (if available) and times it
- Prints nvidia-smi output if available

Run:
    python gpu_test.py

"""

import time
import subprocess

def run_nvidia_smi():
    try:
        print('\n=== nvidia-smi output ===')
        out = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT, universal_newlines=True)
        print(out)
    except Exception as e:
        print('nvidia-smi not available or failed:', e)


def tensorflow_test(size=1024):
    try:
        import tensorflow as tf
        print('\n=== TensorFlow ===')
        gpus = tf.config.list_physical_devices('GPU')
        print('Physical GPUs:', gpus)
        if gpus:
            try:
                # Allocate and run a matmul on GPU: use a slightly larger size to see measurable time
                a = tf.random.uniform((size, size), dtype=tf.float32)
                b = tf.random.uniform((size, size), dtype=tf.float32)
                # Force placement on GPU
                with tf.device('/GPU:0'):
                    t0 = time.time()
                    c = tf.linalg.matmul(a, b)
                    # fetch to host
                    _ = c.numpy()
                    t1 = time.time()
                print(f'TensorFlow matmul ({size}x{size}) time: {t1-t0:.4f} s')
            except Exception as e:
                print('TensorFlow GPU matmul failed:', e)
        else:
            print('No GPUs found for TensorFlow')
    except Exception as e:
        print('TensorFlow not available:', e)


def pytorch_test(size=1024):
    try:
        import torch
        print('\n=== PyTorch ===')
        if torch.cuda.is_available():
            dev = torch.device('cuda:0')
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:
                name = 'cuda:0'
            print('CUDA available. Device 0:', name)
            try:
                a = torch.randn(size, size, device=dev, dtype=torch.float32)
                b = torch.randn(size, size, device=dev, dtype=torch.float32)
                # warm-up
                for _ in range(2):
                    _ = torch.matmul(a, b)
                torch.cuda.synchronize()
                t0 = time.time()
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
                t1 = time.time()
                print(f'PyTorch matmul ({size}x{size}) time: {t1-t0:.4f} s')
            except Exception as e:
                print('PyTorch GPU matmul failed:', e)
        else:
            print('PyTorch CUDA not available')
    except Exception as e:
        print('PyTorch not available:', e)


if __name__ == '__main__':
    print('GPU smoke-test')
    run_nvidia_smi()
    # use 1024 or 2048 depending on your GPU memory; default 1024 is conservative
    tensorflow_test(size=1024)
    pytorch_test(size=1024)
    print('\nDone')
