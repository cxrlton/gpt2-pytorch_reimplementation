# gpt2 - pytorch reimplementation


## Training Details
### Optimization:

* Optimizer: AdamW with betas (0.9, 0.95) and an epsilon value of 1e-8.
* Learning Rate Schedule:
  
    * Warmup Steps: 715 steps for linearly ramping up to the maximum learning rate.
    * Cosine Decay: Learning rate decays from the maximum (6e-4) to a minimum (6e-5) over the total training steps.
    * Gradient Clipping: Gradients are clipped at a norm of 1.0 to prevent instability.
    * Gradient Accumulation: Used to simulate large effective batch sizes, with gradients accumulated over multiple micro-batches.
      
* Batch Size:
  * Global Batch Size: 524,288 tokens per update step.
  * Microbatch Size: 16 sequences per GPU.
  * Sequence Length: 1,024 tokens.
* Loss Function: Cross-entropy loss over the logits computed for the target tokens.

## Model Parameters
* Total Parameters(for the basic configuration):
  * ~124M paramaters
 
``` Using device: mps
=== GPT-2 Model Dimensions ===
Vocabulary Size: 50304
Max Sequence Length: 1024
Hidden Size (d_model): 768
Number of Layers: 12
Number of Attention Heads: 12
Head Dimension: 64
Feed Forward Size: 3072
Embedding Size: 768
Context Window: 1024
Total Parameters: 124,475,904
```

## Training Setup
  * Designed for multi-GPU training using Distributed Data Parallel (DDP) with NCCL as the bsckend.
  * Supports CUDA devices, but can work for MPS (Apple Silicon's Metal) /CPU.
  * Used 4x NVIDIA L40S GPUs, with a batch size ```B = 16```.

## Other Optimizations
* ```torch.set_float32_matmul_precision('high')```:  Running float32 matrix multiplications in lower precision has significantly increased
    performance. Expected upto 19.5 TFLOPS of performance.

   Important as the architecture is just a series of matrix multiplications. Biggest matrix multiplcation is actually done at the top, at the classifier layer going from 768 to 50257.
* ```torch.compile``` a method to speed up your PyTorch code! torch.compile makes PyTorch code run faster by JIT-compiling PyTorch code into optimized kernels, all while requiring minimal code changes. The Pytorch compiler looks at the code as a whole, observes the operations the code tends to run, and optimizes it. 
* Flash Attention: kernel fusion 
algorithm (cannot be found by torch.compile as flash attention is an algorithmic re-write). Potentially upto 6x faster than the standard attention mechanism.
<p align="center">
  <img src="https://github.com/user-attachments/assets/39cc0692-74f9-4d20-9ad0-29c1becb127e" width="400" alt="Flash Attention Operation"/>
</p>
* Changed vocab_size : 50257 -> 50304 (divisible by 2<sup>x</sup> ->   x = 1, ..., 7)

## Results
### For tinyshakespeare dataset:

```
Final Training Loss: 0.0373
Final Validation Loss: 0.0372
```

## Datasets Used
* FineWeb-Edu[sample-10B tokens] https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
* tinyshakespeare https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
