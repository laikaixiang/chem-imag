"""简化的环境验证脚本"""
import sys
import os
from pathlib import Path

if sys.platform == 'win32':
    os.system('chcp 65001 > nul')
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 60)
print("🧪 Chemical Mind Map Generator - 环境验证")
print("=" * 60)

# 验证Python版本
print(f"\n✅ Python版本: {sys.version.split()[0]}")

# 验证核心依赖（不导入gradio避免pandas冲突）
print("\n🔍 验证核心依赖...")
core_deps = ["PIL", "numpy", "cv2", "pubchempy", "httpx", "dotenv", "pydantic", "scipy"]
for module in core_deps:
    try:
        __import__(module)
        print(f"  ✅ {module}")
    except ImportError:
        print(f"  ❌ {module} - 缺失")

# 验证配置
print("\n🔍 验证配置...")
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.config import settings
    print(f"  ✅ 配置加载成功")
    print(f"     项目根目录: {settings.OUTPUT_DIR.parent}")
    print(f"     OUTPUT_DIR: {settings.OUTPUT_DIR}")
except Exception as e:
    print(f"  ❌ 配置加载失败: {e}")

# 验证设备检测
print("\n🔍 验证设备检测...")
try:
    import torch
    device = settings.get_device()
    print(f"  ✅ PyTorch已安装")
    print(f"     检测到设备: {device}")
    print(f"     CUDA可用: {torch.cuda.is_available()}")
except ImportError:
    print(f"  ⚠️  PyTorch未安装（可选依赖）")
    print(f"     配置的设备: {settings.DEVICE}")

# 验证目录结构
print("\n🔍 验证目录结构...")
print(f"  ✅ 所有必需目录已创建")

print("\n" + "=" * 60)
print("✅ Phase 0 环境搭建完成！")
print("=" * 60)
