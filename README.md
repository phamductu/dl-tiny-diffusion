# tiny-diffusion
A simple Pytorch implementation of probabilistic diffusion models. The starting point is [tiny-diffusion](https://github.com/tanelp/tiny-diffusion) for 2D datasets. Then, it is adapted to run for MNIST and CIFAR-10.

Get started by running `python main.py -h` to explore the available options for training.

For MNIST and CIFAR-10, run `python gen.py -h` to explore the available options for visualization.

## References

* The dino dataset comes from the [Datasaurus Dozen](https://www.autodesk.com/research/publications/same-stats-different-graphs) data.
* [tiny-diffusion](https://github.com/tanelp/tiny-diffusion).
* Jonathan Ho's [implementation of DDPM](https://github.com/hojonathanho/diffusion).
* John Calzeretta's [implementation of DDPM for MNIST](https://github.com/jcalz23/diffusion_diy).

