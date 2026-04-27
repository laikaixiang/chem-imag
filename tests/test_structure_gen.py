"""Phase 1 测试：StructureGenerator"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.structure_gen.generator import StructureGenerator, STYLE_PRESETS

passed = 0
failed = 0


def run_test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  ✅ {name}")
        passed += 1
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        traceback.print_exc()
        failed += 1


def test_aspirin_smiles():
    gen = StructureGenerator()
    path, img = gen.generate_from_smiles(
        "CC(=O)Oc1ccccc1C(=O)O",
        output_path=gen.output_dir / "aspirin_acs1996.png",
    )
    # path, img = gen.generate_from_smiles(
    #     "c1ccc(-n2c3ccccc3c3c2c2c4ccccc4n(-c4ccc(-c5ccc(B6c7ccccc7B(c7ccccc7-c7ccc(-n8c9ccccc9c9c%10c(c%11ccccc%11n%10-c%10ccccc%10)c%10c(c%11ccccc%11n%10-c%10ccccc%10)c98)cc7)c7cccc(-c8ccc(-n9c%10ccccc%10c%10c%11c(c%12ccccc%12n%11-c%11ccccc%11)c%11c(c%12ccccc%12n%11-c%11ccccc%11)c%109)cc8)c76)c(-c6ccc(-n7c8ccccc8c8c9c(c%10ccccc%10n9-c9ccccc9)c9c(c%10ccccc%10n9-c9ccccc9)c87)cc6)c5)cc4)c2c2c4ccccc4n(-c4ccccc4)c32)cc1",
    #     output_path=gen.output_dir / "aspirin_acs1996.png",
    # )
    assert path.exists(), f"文件未生成: {path}"
    assert img.size[0] > 0 and img.size[1] > 0
    print(f"    -> {path}")


def test_caffeine_by_name():
    gen = StructureGenerator(default_style="dark_mode")
    path, img = gen.generate_from_name(
        "caffeine",
        output_path=gen.output_dir / "caffeine_dark.png",
    )
    assert path.exists(), f"文件未生成: {path}"
    print(f"    -> {path}")


def test_four_styles():
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    gen = StructureGenerator()
    for style_name in STYLE_PRESETS:
        path, img = gen.generate_from_smiles(
            smiles,
            output_path=gen.output_dir / f"aspirin_{style_name.lower()}.png",
            style=style_name,
        )
        assert path.exists(), f"风格 {style_name} 文件未生成"
    print(f"    -> 4 种风格全部生成")


def test_transparent_background():
    gen = StructureGenerator()
    _, img = gen.generate_from_smiles("CC(=O)Oc1ccccc1C(=O)O")
    assert img.mode == "RGBA", f"期望 RGBA，实际 {img.mode}"
    alpha = img.getchannel("A")
    pixels = list(alpha.getdata())
    zero_count = pixels.count(0)
    assert zero_count > 0, "透明背景应有 alpha=0 的像素"
    print(f"    -> RGBA 模式，透明像素数: {zero_count}")


def test_invalid_smiles():
    gen = StructureGenerator()
    try:
        gen.generate_from_smiles("NOT_A_SMILES!!!")
        assert False, "应抛出 ValueError"
    except ValueError:
        pass


def test_horizontal_layout():
    gen = StructureGenerator()
    compounds = [
        {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "label": "Aspirin"},
        {"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "label": "Caffeine"},
    ]
    path, img = gen.generate_multiple(
        compounds,
        layout="horizontal",
        output_path=gen.output_dir / "caffeine_aspirin_horizontal.png",
    )
    assert path.exists()
    assert img.size[0] > img.size[1], "水平布局宽度应大于高度"
    print(f"    -> {path} ({img.size[0]}x{img.size[1]})")


def test_grid_layout():
    gen = StructureGenerator()
    compounds = [
        {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "label": "Aspirin"},
        {"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "label": "Caffeine"},
        {"smiles": "CC(=O)O", "label": "Acetic Acid"},
        {"smiles": "C1=CC=CC=C1", "label": "Benzene"},
    ]
    path, img = gen.generate_multiple(
        compounds,
        layout="grid",
        output_path=gen.output_dir / "grid_4compounds.png",
    )
    assert path.exists()
    print(f"    -> {path} ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    print("Phase 1 测试: StructureGenerator\n")

    run_test("1. 阿司匹林 SMILES 生成", test_aspirin_smiles)
    run_test("2. 咖啡因名称查询生成", test_caffeine_by_name)
    run_test("3. 四种风格输出", test_four_styles)
    run_test("4. 透明背景验证", test_transparent_background)
    run_test("5. 无效 SMILES 异常", test_invalid_smiles)
    run_test("6. 水平布局多分子", test_horizontal_layout)
    run_test("7. 网格布局", test_grid_layout)

    print(f"\n结果: {passed}/{passed + failed} 通过")
    if failed == 0:
        print("🎉 全部通过！Phase 1 完成。")
    else:
        print(f"⚠️ {failed} 个测试失败")
        sys.exit(1)
