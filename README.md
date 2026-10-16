# ControlFlow3D

Official repository for **ControlFlow3D: Distilling Multi-View Knowledge into Latent Flow Matching for Point Cloud Upsampling**.

**Paper:** [arXiv:2610.21190](https://arxiv.org/abs/2610.21190)

> **Code coming soon.** The training and inference code will be released in this repository. Stay tuned!

## ShapeNetPU Dataset

**[Download ShapeNetPU and read the dataset documentation on Hugging Face](https://huggingface.co/datasets/BaldFatTiger/ShapeNetPU)**

ShapeNetPU is a ShapeNet-derived dataset for **point cloud upsampling**, covering **13 object categories**.

- **35,762 shapes:** 30,395 for training and 5,367 for testing.
- **Paired point clouds:** 2,048-point sparse inputs and 8,192-point dense ground truth.
- **Multi-view training data:** eight rendered views per training shape, with view lists and rendering metadata.
- **Evaluation geometry:** reference meshes for the test split.

The [dataset card](https://huggingface.co/datasets/BaldFatTiger/ShapeNetPU#readme) provides the directory structure, category-level splits, download instructions, file-format details, and loading examples.

## Citation

If you find ControlFlow3D or ShapeNetPU useful in your research, please cite:

```bibtex
@misc{liu2026controlflow3d,
  title={ControlFlow3D: Distilling Multi-View Knowledge into Latent Flow Matching for Point Cloud Upsampling},
  author={Yuang Liu and Zhi Zuo and Zhengkai Zhao and Lirui Zhang and Pan Gao and Hao Feng and Zhengzhe Liu},
  year={2026},
  eprint={2610.21190},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.21190},
}
```

## Terms of Use

ShapeNetPU is derived from ShapeNet. Please follow the [dataset terms of use](https://huggingface.co/datasets/BaldFatTiger/ShapeNetPU#license-and-terms-of-use) and the applicable upstream licenses.
