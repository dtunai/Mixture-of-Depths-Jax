## Mixture of Depths (Jax/Flax)

Jax + Flax implementation routing mechanism of the paper [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258). Architecture enforces a total compute budget by capping the number of tokens (𝑘) that can participate in the self-attention and MLP computations at a given layer. The tokens to be processed are determined by the network using a top-𝑘 routing mechanism.

### Key Information About Architecture

- Dynamic Compute Allocation: Demonstrates that transformer-based language models can dynamically allocate computational resources (FLOPs) to specific positions in a sequence, optimizing the allocation across different layers and model depths.

- Top-K Routing Mechanism:  Method enforces a total compute budget by capping the number of tokens that can participate in computations at a given layer. It uses a top-k routing mechanism to select tokens to be processed, resulting in a static computation graph with known tensor sizes while maintaining the flexibility to adjust which tokens are processed.

- Efficiency and Performance: Models trained using this dynamic compute allocation approach achieve baseline performance while using significantly fewer FLOPs per forward pass, resulting in up to 50% faster post-training sampling times.

### TODOs

- [] Usage and training code will be added to the repository.
- [] Attention block, RMSNorm block, FFN block will be implemented inside the MoD routing layer.
