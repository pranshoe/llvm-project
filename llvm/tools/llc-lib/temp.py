import ctypes
import os
import tempfile

lib_path = os.path.expanduser("~/llvm-project/build/lib/libLLVMLLCDriver.so")
lib_path = os.path.abspath(lib_path)  # Ensure fully resolved

lib = ctypes.CDLL(lib_path)

with open("test.ll", 'r') as f:
    f_bytes = f.read().encode('utf-8')

run_opt = lib.main
args = [b"llc", b"-mtriple=i686-unknonw-unknown", b"-mattr=+sse2"]
argc = len(args)

argv_array_type = ctypes.c_char_p * argc
argv = argv_array_type(*args)

pipe_r, pipe_w = os.pipe()
orig_stdin = os.dup(0)
with tempfile.TemporaryFile() as temp_input:
    temp_input.write(f_bytes)

    temp_input.seek(0)

    os.dup2(temp_input.fileno(), 0)
    exit_code = lib.main(argc, argv)


os.dup2(orig_stdin, 0)
os.close(orig_stdin)



