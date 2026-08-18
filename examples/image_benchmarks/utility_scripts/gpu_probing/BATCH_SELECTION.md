# Batch-size selection — summary

Host: escher-02 · 8x A100-40GB · conda env `neuripp_cuda13`
Protocol: `PROBING_PROTOCOL.md` · Tool: `probe_ngd_memory.py` (NGD worst-case path, `--warmup 8 --measure 20`, `PREALLOCATE=false`)

## Selected batch sizes

| dataset | arch | GPUs | global batch | per-GPU batch | peak GiB | s/step | ms/image | SM% mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| afhq_cat | latent-UNet @256 | 2 | 256 | 128 | 19.2 | 75.91 | 296.53 | 99.93 |
| afhq_cat | latent-UNet @256 | 4 | 512 | 128 | 19.2 | 75.97 | 148.38 | 99.92 |
| cifar10 | UNet CIFAR-reference | 1 | 50 | 50 | 18.5 | 115.48 | 2309.68 | 99.94 |
| cifar10 | UNet small | 1 | 100 | 100 | 9.7 | 59.33 | 593.32 | 99.85 |
| fashion_mnist | UNet small | 1 | 2000 | 2000 | 17.4 | 43.94 | 21.97 | 98.96 |
| ffhq64 | UNet CIFAR-reference @64 | 1 | 50 | 50 | 31.1 | 440.65 | 8812.96 | 99.97 |
| ffhq64 | UNet CIFAR-reference @64 | 2 | 50 | 25 | 19.1 | 221.57 | 4431.33 | 99.95 |
| flowers102 | SiT-S/2 @256 | 2 | 32 | 16 | 19.1 | 5.86 | 182.97 | 99.05 |
| flowers102 | SiT-S/2 @256 | 4 | 64 | 16 | 19.1 | 5.88 | 91.8 | 99.15 |
| flowers102 | UNet small @64 | 1 | 64 | 64 | 17.7 | 144.8 | 2262.56 | 99.79 |
| flowers102 | UNet small @64 | 2 | 128 | 64 | 18.4 | 144.89 | 1131.96 | 99.59 |
| flowers102 | latent-UNet @256 | 2 | 256 | 128 | 19.2 | 75.91 | 296.54 | 99.89 |
| flowers102 | latent-UNet @256 | 4 | 512 | 128 | 19.2 | 75.99 | 148.42 | 99.92 |
| imagenet64 | SiT-S/2 @64 | 4 | 32 | 8 | 31.7 | 15.01 | 468.95 | 99.58 |
| imagenet64 | UNet small @64 | 1 | 64 | 64 | 17.7 | 144.61 | 2259.47 | 99.95 |
| imagenet64 | UNet small @64 | 2 | 128 | 64 | 18.4 | 144.88 | 1131.89 | 99.93 |
| lsun_church | SiT-S/2 @256 | 2 | 32 | 16 | 19.1 | 5.85 | 182.71 | 99.18 |
| lsun_church | SiT-S/2 @256 | 4 | 64 | 16 | 19.1 | 5.88 | 91.81 | 99.14 |
| lsun_church | latent-UNet @256 | 2 | 256 | 128 | 19.2 | 75.87 | 296.37 | 99.93 |
| lsun_church | latent-UNet @256 | 4 | 512 | 128 | 19.2 | 75.89 | 148.23 | 99.92 |
| mnist | MLP | 1 | 3000 | 3000 | 1.7 | 0.2 | 0.07 | 24.06 |
| mnist | MLP on AE-32 latent | 1 | 3000 | 3000 | 1.6 | 0.11 | 0.04 | 94.0 |
| mnist | MLP on AE-64 latent | 1 | 3000 | 3000 | 1.6 | 0.11 | 0.04 | 93.29 |
