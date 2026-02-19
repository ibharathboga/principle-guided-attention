
import torch
import time

def benchmark_svd():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Benchmarking on {device}")
    
    B = 32
    C = 64
    
    # Create random batch of (B, C, C) - max size for T=64
    x = torch.randn(B, C, C, device=device)
    
    start = time.time()
    for _ in range(100):
        # SVD
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)
        # P computation
        target_rank = C // 2
        V_top = Vh[:, :target_rank, :]
        P = V_top.transpose(1, 2) @ V_top
        torch.cuda.synchronize() if device == 'cuda' else None
        
    end = time.time()
    avg_time = (end - start) / 100
    print(f"Average time per SVD batch (B=32, C=64): {avg_time*1000:.2f} ms")
    print(f"Estimated time per step (T=64, Stride=4 => 16 SVDs): {avg_time * 16 * 1000:.2f} ms")
    
if __name__ == "__main__":
    benchmark_svd()
