#!/usr/bin/env bash
set -euo pipefail
# NVIDIA PeopleNet — NGC / TAO / TensorRT (.etlt), не PyTorch one-shot eval.

echo "=== PeopleNet (группа B, слот 8) ==="
echo "Каталог NGC: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet"
echo "Типичный путь: TAO train/export → TensorRT engine на Jetson или dGPU."
echo ""
echo "Зафиксируйте FPS и точность на вашем пайплайне, затем сохраните JSON:"
echo "  model=peoplenet_resnet34  group=B  detector_id=8"
echo ""
echo "Пример merge метаданных в уже созданный JSON:"
echo "  python3 scripts/bench_runner.py --merge-json results/runs/peoplenet_....json \\"
echo "    --patch-json docs/templates/group_b_meta_patch.example.json"
echo "  (отредактируйте пример: добавьте metrics из вашего отчёта TAO)"
