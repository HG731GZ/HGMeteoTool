# MeteoAlign

面向星野摄影与流星雨摄影的离线星空视野模拟工具。

## 配置环境

推荐使用项目自带的 `environment.yml` 创建 conda 环境：

```bash
conda env create -f environment.yml
conda activate hgastro
```

如果环境已经存在，可以更新：

```bash
conda env update -n hgastro -f environment.yml
conda activate hgastro
```

可选：检查环境是否正确：

```bash
python scripts/verify_hgastro_env.py
```

## 运行程序

在项目根目录运行：

```bash
python main.py
```

也可以使用包启动方式：

```bash
python -m meteoalign
```

当前程序依赖 `catalog/` 目录下的离线星表数据，请保持 `catalog` 文件夹与源码一起放在项目根目录。
