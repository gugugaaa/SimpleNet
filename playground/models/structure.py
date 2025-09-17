"""
查看模型结构

例如，efficientnetv2_m的结构是
model
├── conv_stem
├── bn1
├── blocks (Sequential)
│   ├── blocks.0 (Block)
│   ├── blocks.1 (Block)
│   ├── ...
│   └── blocks.6 (Block)
├── conv_head
├── bn2
├── global_pool
├── classifier
"""

import torch

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backbones import load

# print(torch.hub.get_dir()) 

model = load("resnet50")  # 或 efficientnetv2_m, wideresnet50, resnet18
model.eval()
probe = torch.randn(1,3,224,224)

feat_shapes = {}
hooks = []
def reg(name, m):
    def fn(_, __, out):
        if isinstance(out, torch.Tensor):
            feat_shapes[name] = tuple(out.shape)
    return m.register_forward_hook(fn)

# 顶层
for name, module in list(model._modules.items()):
    # blocks 是 Sequential，内部再分 index
    if name == "blocks":
        for i, blk in enumerate(module):
            hooks.append(reg(f"blocks.{i}", blk))
    else:
        hooks.append(reg(name, module))

with torch.no_grad():
    _ = model(probe)

for k in sorted(feat_shapes.keys()):
    print(k, feat_shapes[k])

for h in hooks: h.remove()