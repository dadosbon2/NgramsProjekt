import torch


def check_cuda():
    print("Checking CUDA and GPU availability...\n")

    # Check if CUDA is available
    if torch.cuda.is_available():
        print("✅ CUDA is available!")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB")
            print(f"Memory Cached: {torch.cuda.memory_reserved(i) / 1024 ** 3:.2f} GB")

    else:
        print("❌ CUDA is NOT available. Running on CPU.")


if __name__ == "__main__":
    check_cuda()
