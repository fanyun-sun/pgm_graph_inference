# Equivariant neural network for factor graphs

*Authors: Fan-Yun Sun, Jonathan Kuck, Hao Tang, Stefano Ermon

[Link to the paper](https://arxiv.org/abs/2109.14218)

##

* To generate data: `$ ./make_data.sh grid`
* For the list of available models, refer to `inference/__init.py`
* To run experiments: `$./run_all.sh grid MODEL TRAIN_NUM`

## Citation

```
@article{sun2021equivariant,
  title={Equivariant neural network for factor graphs},
  author={Sun, Fan-Yun and Kuck, Jonathan and Tang, Hao and Ermon, Stefano},
  journal={arXiv preprint arXiv:2109.14218},
  year={2021}
}
```

We thank the following work: [Inference in graphical models with GNNs
](https://github.com/krishvishal/pgm_graph_inference/tree/master)
