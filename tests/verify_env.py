"""验证环境依赖"""
import sys
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')
    sys.stdout.reconfigure(encoding='utf-8')

def verify_dependencies():
    print("🔍 验证依赖包...")

    required = {
        "PIL": "Pillow",
        "numpy": "numpy",
        "cv2": "opencv-python-headless",
        "pubchempy": "pubchempy",
        "httpx": "httpx",
        "dotenv": "python-dotenv",
        "pydantic": "pydantic",
        "gradio": "gradio",
        "scipy": "scipy",
    }

    optional = {
        "torch": "torch (optional-detection)",
        "torchvision": "torchvision (optional-detection)",
        "transformers": "transformers (optional-detection)",
        "rwkv": "rwkv (optional-rwkv)",
    }

    missing = []
    installed = []

    for module, package in required.items():
        try:
            __import__(module)
            installed.append(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"  ❌ {package} - 缺失")

    print("\n🔍 验证可选依赖...")
    for module, package in optional.items():
        try:
            __import__(module)
            installed.append(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ⚠️  {package} - 未安装（可选）")

    if missing:
        print(f"\n❌ 缺少必需依赖: {', '.join(missing)}")
        print("请运行: pip install -e .")
        return False

    print("\n✅ 所有必需依赖已安装")
    return True

def verify_config():
    print("\n🔍 验证配置...")
    try:
        from src.config import settings
        print(f"  ✅ 配置加载成功")
        print(f"     ROOT: {settings.OUTPUT_DIR.parent}")
        print(f"     OUTPUT_DIR: {settings.OUTPUT_DIR}")
        print(f"     TEMP_DIR: {settings.TEMP_DIR}")

        try:
            device = settings.get_device()
            print(f"     设备: {device}")
        except ImportError:
            print(f"     设备: {settings.DEVICE} (torch未安装，无法自动检测)")

        return True
    except Exception as e:
        print(f"  ❌ 配置加载失败: {e}")
        return False

def verify_structure():
    print("\n🔍 验证目录结构...")
    root = Path(__file__).resolve().parent.parent

    required_dirs = [
        "src",
        "src/structure_gen",
        "src/compositor",
        "src/mindmap",
        "src/scene_gen",
        "src/agent",
        "src/gui",
        "tests",
        "outputs",
        "outputs/structures",
        "outputs/scenes",
        "outputs/mindmaps",
        "outputs/final",
        "assets",
        "assets/templates",
    ]

    missing_dirs = []
    for d in required_dirs:
        path = root / d
        if path.exists():
            print(f"  ✅ {d}/")
        else:
            missing_dirs.append(d)
            print(f"  ❌ {d}/ - 缺失")

    if missing_dirs:
        print(f"\n❌ 缺少目录: {', '.join(missing_dirs)}")
        return False

    print("\n✅ 目录结构完整")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Chemical Mind Map Generator - 环境验证")
    print("=" * 60)

    deps_ok = verify_dependencies()
    config_ok = verify_config()
    struct_ok = verify_structure()

    print("\n" + "=" * 60)
    if deps_ok and config_ok and struct_ok:
        print("✅ 环境验证通过！")
        sys.exit(0)
    else:
        print("❌ 环境验证失败，请检查上述错误")
        sys.exit(1)
