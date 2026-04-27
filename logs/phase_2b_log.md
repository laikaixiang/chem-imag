# Phase 2b: 光照匹配与纹理融合实现日志

**作者**: lkx  
**日期**: 2026-04-27  
**状态**: 完成

## 目标

实现 `compositor/lighting.py` 模块，消除合成图像的"PS感"：
- Reinhard 色彩迁移（Lab空间 + 均值/标准差匹配）
- 高频纹理提取与叠加
- 投影生成与应用
- 一体化 match_and_blend 接口

## 实现内容

### 1. 创建 `src/compositor/lighting.py` (180行)

实现了 6 个核心函数：

#### `color_transfer(src, target, mask=None) -> np.ndarray`
- Reinhard 色彩迁移算法（简化版）
- 在 Lab 色彩空间进行均值-标准差匹配
- 支持 mask，仅对前景有效区域计算统计量
- 公式: `result = (src - src_mean) * (target_std / src_std) + target_mean`

#### `extract_texture(image, sigma=8.0) -> np.ndarray`
- 提取图像表面高频纹理成分
- 使用 GaussianBlur 分离低频，保留高频
- 纹理值限制在 [-30, 30] 范围

#### `apply_texture(struct, texture, strength=0.1) -> np.ndarray`
- 将提取的纹理微量叠加到结构图上
- 使用默认 0.1 强度，避免喧宾夺主

#### `generate_shadow(alpha, offset=(3,3), blur=5, opacity=0.25) -> np.ndarray`
- 根据 alpha mask 生成柔和投影
- 支持自定义偏移、模糊半径和不透明度
- 返回 float32 单通道投影图

#### `apply_shadow(background, shadow, position) -> np.ndarray`
- 将投影叠加到背景上（通过暗化实现）
- 公式: `darkened = bg_region * (1 - shadow)`

#### `match_and_blend(background, foreground, position, color_match, texture_blend, shadow, feather) -> np.ndarray`
- 一体化接口，组合所有步骤
- 完整流程：feather → color transfer → texture blend → shadow → alpha composite
- 对外的主要调用接口

### 2. 更新 `src/compositor/__init__.py`

导出 6 个新函数：color_transfer, extract_texture, apply_texture, generate_shadow, apply_shadow, match_and_blend

### 3. 创建测试 `tests/test_compositor.py`

实现了 8 个测试用例：
1. **Color Transfer** - 测试基础色彩迁移
2. **Color Transfer with Mask** - 测试带 mask 的色彩迁移
3. **Texture Extraction** - 测试纹理提取
4. **Texture Application** - 测试纹理叠加
5. **Shadow Generation** - 测试投影生成
6. **Shadow Application** - 测试投影应用到背景
7. **Match and Blend** - 测试一体化接口
8. **Plain vs Match-and-Blend** - 对比未处理 vs 光照匹配效果

## 测试结果

```bash
python tests/test_compositor.py
```

所有 8 个测试通过
输出保存在: `outputs/compositor/`

生成的测试图像：
- `test_color_transfer.png` - 色彩迁移效果
- `test_shadow_map.png` - 投影灰度图
- `test_match_and_blend.png` - 一体化合成效果
- `test_comparison_plain.png` - 普通 alpha 合成（对照组）
- `test_comparison_matched.png` - 光照匹配合成（实验组）

## 技术要点

### Reinhard 色彩迁移
```python
src_lab = cv2.cvtColor(src, cv2.COLOR_RGB2Lab).astype(np.float32)
target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2Lab).astype(np.float32)
result = (src_lab - src_mean) * (tgt_std / src_std) + tgt_mean
```

### 纹理融合
```python
blurred = cv2.GaussianBlur(image_f, (0, 0), sigma)
texture = image_f - blurred  # 高频 = 原图 - 低频
result = struct + texture * strength
```

### 投影暗化
```python
darkened = bg_region * (1 - shadow)  # shadow值越大越暗
```

## 文件清单

**新增文件**:
- `src/compositor/lighting.py` (180 行)

**修改文件**:
- `src/compositor/__init__.py` - 添加 6 个新函数导出
- `tests/test_compositor.py` - 添加 8 个光照匹配测试
- `CLAUDE.md` - 更新 Phase 2b 状态为完成

## 下一步

Phase 2c: 实现拉普拉斯金字塔融合
- 高斯金字塔构建
- 拉普拉斯金字塔构建
- 金字塔混合
- 图像重建
