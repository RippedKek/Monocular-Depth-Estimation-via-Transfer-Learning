## Experiments Ran

As this paper was experimented on huge datasets like NYU Dataset, we took a chunk of it and ran the experiment for 5 epochs, batch size 4. The experiment was ran on RTX 4050 laptop GPU. This is the config for the first run. The dataset only had images of classrooms rather than various indoor scenaries. After this run, we will run some ablation studies on this codebase.

**Note that this codebase is being run on PyTorch. So train the dataset from the `PyTorch` directory**

---

## How to Run

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cd PyTorch
python train.py --bs 4 --epochs 5
```

`--bs` corresponds to batch size. For limited hardware capabilities, we ran on batch size 4.

## Reference

Corresponding paper to cite:

```
@article{Alhashim2018,
  author    = {Ibraheem Alhashim and Peter Wonka},
  title     = {High Quality Monocular Depth Estimation via Transfer Learning},
  journal   = {arXiv e-prints},
  volume    = {abs/1812.11941},
  year      = {2018},
  url       = {https://arxiv.org/abs/1812.11941},
  eid       = {arXiv:1812.11941},
  eprint    = {1812.11941}
}
```
