# 三维动态接取场景

固定底座的 xArm7 配备 LEAP 右手，共 23 个驱动关节。机械臂保持初始姿态，物体沿瞄准张开手掌的轨迹飞行。场景用于调试拦截前的轨迹、接触和扰动。首次接触只表示手碰到了物体。主动拦截、闭指和稳定抓持判定属于后续控制模块。

从仓库根目录运行（Windows 或 Linux，使用已激活的 Python 环境）：

```sh
python -m pip install -r scene_setup/requirements.txt
python scene_setup/scene.py
python scene_setup/scene.py --object box --gust --seed 1
python scene_setup/scene.py --object ball --headless
```

默认橙色球的半径为 25 mm。长方体为 64 × 42 × 34 mm，带初始自转。两种物体均为 60 g，自由运动，惯量由几何形状与质量确定。启用 `--gust` 时，仿真按种子生成一段恒定世界坐标系加速度，用外力施加。关闭窗口结束演示，`--headless` 则运行 1.5 秒并打印首次手部接触时间和最终位置（SI 单位）。每次重新启动脚本即可重放。

模型和腕部装配参考 [mink 示例](https://github.com/kevinzakka/mink/blob/5f800f0a369d05bab88b5e5582cf95122d3b82a5/examples/arm_hand_xarm_leap.py)。所需 XML 和网格已随仓库提供。各模型授权见 `models/ufactory_xarm7/LICENSE` 和 `models/leap_hand/LICENSE`，装配示例许可见 `models/MINK_LICENSE`。
