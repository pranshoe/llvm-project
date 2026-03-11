import ctypes
import os

lib_path = os.path.expanduser("~/llvm-project/build/lib/libLLVMOptDriver.so")
lib_path = os.path.abspath(lib_path)  # Ensure fully resolved

lib = ctypes.CDLL(lib_path)

run_opt = lib.main
args = [b"opt", b"-passes=loop-vectorize", b"-mtriple=aarch64", b"-mattr=+crypto", b"-S", b"test.ll"]
argc = len(args)

argv_array_type = ctypes.c_char_p * argc
argv = argv_array_type(*args)

exit_code = lib.main(argc, argv)

print(exit_code)

