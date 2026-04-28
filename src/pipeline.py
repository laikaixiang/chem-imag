"""E2E pipeline: text input -> mindmap image output.

Compound extraction is LLM-driven: the user's natural-language input is
sent to a lightweight model that returns a structured compound list + tree.
Falls back to a minimal default list when no LLM is available.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from src.config import api_config, settings

logger = logging.getLogger(__name__)

# Prompt sent to the LLM for compound extraction
_EXTRACT_PROMPT = """Extract all organic chemistry compounds from the user's description below.
Return a JSON object with this exact structure (no other text):

{{
  "title": "short mindmap title (in English)",
  "compounds": [
    {{
      "name": "IUPAC name in English (e.g. phenol, ethanol, benzoic acid)",
      "smiles": "canonical SMILES string for this compound",
      "parent": "parent compound English name or null for top-level"
    }}
  ]
}}

CRITICAL RULES:
- ALL compound names MUST be in English (IUPAC). Never use Chinese names.
- Provide the canonical SMILES string for every compound.
- If the user mentions a reaction, include both reactants and products.
- If the user mentions a compound class (e.g. "alcohols"), expand to 2-3 representative examples.

User description:
{user_input}

JSON:"""


class ChemicalImagePipeline:
    """End-to-end pipeline from natural language to final composite image.

    Usage:
        pipe = ChemicalImagePipeline()
        result = pipe.generate("生成关于醇类氧化反应的思维导图")
        print(result["final_image"])
    """

    def __init__(self, output_dir: Optional[Path] = None, api_key: Optional[str] = None,
                 image_provider: str = "packyapi"):
        self.output_dir = output_dir or settings.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._api_key = api_key or api_config.key
        self._image_provider = image_provider

    # ── public API ──────────────────────────────────────────────

    def generate(
        self,
        user_input: str,
        style: str = "academic",
        width: int = 1920,
        height: int = 1080,
        use_agent: bool = False,
        use_optimizer: bool = False,
    ) -> dict:
        """Run the full pipeline and return result metadata.

        Args:
            user_input: Natural-language description.
            style: Scene style — "academic", "modern", or "minimal".
            width, height: Output image dimensions.
            use_agent: If True, delegate to AgentOrchestrator (needs LLM API).
            use_optimizer: If True, call LLM to optimize the prompt first.

        Returns:
            {"final_image": path, "steps": [...], "compounds": [...]}
        """
        if use_agent:
            return self._run_agent(user_input, style)

        steps: list[str] = []

        # Step 0 (optional): Optimize the user prompt via LLM
        if use_optimizer:
            from src.prompt_optimizer import optimize_prompt

            optimized = optimize_prompt(user_input)
            if optimized != user_input:
                logger.info("Step 0 — prompt optimized")
                steps.append("optimize")
            user_input = optimized

        # Step 1: Parse compounds via LLM
        parsed = self._extract_compounds(user_input)
        compounds = parsed["compounds"]
        title = parsed.get("title", "Organic Compounds")
        logger.info("Step 1 — LLM parsed %d compounds: %s", len(compounds),
                    [c["name"] for c in compounds])

        # Step 2: Resolve each compound to SMILES (use LLM-provided SMILES when available)
        resolved = self._resolve_compounds(compounds)
        steps.append("resolve")
        logger.info("Step 2 — resolved %d compounds", len(resolved))

        # Step 3: Generate structure images
        struct_results = self._generate_structures(resolved, style)
        steps.append("structures")
        logger.info("Step 3 — generated %d structures", len(struct_results))

        # Step 4: Build mindmap (using tree from LLM)
        mindmap_path = self._build_mindmap_from_parsed(parsed, resolved)
        steps.append("mindmap")
        logger.info("Step 4 — mindmap: %s", mindmap_path)

        # Step 5: Generate background scene
        scene_path = self._generate_scene(prompt=user_input, style=style, width=width, height=height)
        steps.append("scene")
        logger.info("Step 5 — scene: %s", scene_path)

        # Step 6: Composite
        final_path = self._composite(scene_path, struct_results, width, height)
        steps.append("composite")
        logger.info("Step 6 — final: %s", final_path)

        return {
            "final_image": final_path,
            "steps": steps,
            "title": title,
            "compounds": [r["name"] for r in resolved],
            "resolved": resolved,
            "mindmap_path": mindmap_path,
            "scene_path": scene_path,
        }

    # ── agent mode ──────────────────────────────────────────────

    def _run_agent(self, user_input: str, style: str) -> dict:
        from src.agent.orchestrator import AgentOrchestrator

        orch = AgentOrchestrator(llm_provider="default")
        result = orch.run(user_input)
        return {
            "final_image": result.get("final_image", ""),
            "steps": ["agent"],
            "compounds": [],
            "agent_result": result,
        }

    # ── step 1: LLM compound extraction ─────────────────────────

    def _extract_compounds(self, text: str) -> dict:
        """Use an LLM to extract compound names and tree structure from text.

        Returns: {"title": str, "compounds": [{"name": str, "parent": str|null}]}
        Falls back to a minimal default list when LLM is unavailable.
        """
        result = self._call_llm_for_extraction(text)
        if result is not None:
            return result

        # Fallback: minimal default for testing without API key
        logger.info("LLM unavailable — using default compound list")
        return {
            "title": "Organic Compounds",
            "compounds": [
                {"name": "phenol", "parent": None},
                {"name": "benzoic acid", "parent": None},
                {"name": "ethanol", "parent": None},
            ],
        }

    def _call_llm_for_extraction(self, user_input: str) -> Optional[dict]:
        """Call LLM API for compound extraction. Returns None on failure.

        Supports both Anthropic-native and OpenAI-compatible providers
        (SiliconFlow, OpenAI, etc.) determined by api_config.active_provider.
        """
        if not self._api_key:
            return None

        prompt = _EXTRACT_PROMPT.format(user_input=user_input)
        provider = api_config.active_provider

        try:
            if provider == "anthropic":
                return self._call_anthropic(prompt)
            else:
                return self._call_openai_compatible(prompt)
        except Exception as e:
            logger.warning("LLM extraction failed: %s", e)
            return None

    def _call_anthropic(self, prompt: str) -> Optional[dict]:
        import anthropic
        client = anthropic.Anthropic(api_key=self._api_key)
        model = api_config.model("haiku") or "claude-haiku-4-5-20251001"
        resp = client.messages.create(
            model=model, max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_llm_json(resp.content[0].text)

    def _call_openai_compatible(self, prompt: str) -> Optional[dict]:
        import requests
        url = api_config.url or api_config.base_url + "/v1/chat/completions"
        model = api_config.model("talk") or "gpt-4o"
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": 0.1,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return _parse_llm_json(text)

    # ── step 2: resolve compounds ───────────────────────────────

    def _resolve_compounds(self, compounds: list[dict]) -> list[dict]:
        """Resolve compounds to SMILES. Uses LLM-provided SMILES when available."""
        from rdkit import Chem
        from src.structure_gen.generator import StructureGenerator

        gen = StructureGenerator()
        results = []
        for c in compounds:
            name = c.get("name", str(c)) if isinstance(c, dict) else str(c)
            llm_smiles = c.get("smiles", "") if isinstance(c, dict) else ""

            # If LLM provided a valid SMILES, use it directly (validate via RDKit)
            if llm_smiles:
                mol = Chem.MolFromSmiles(llm_smiles)
                if mol is not None:
                    canonical = Chem.MolToSmiles(mol)
                    results.append({"name": name, "smiles": canonical, "status": "ok"})
                    continue
                logger.warning("LLM SMILES invalid for '%s': %s — falling back to PubChem", name, llm_smiles)

            # Fall back to PubChem resolution
            try:
                smiles = gen.resolve(name)
                results.append({"name": name, "smiles": smiles, "status": "ok"})
            except Exception as e:
                logger.warning("resolve failed for '%s': %s", name, e)
                results.append({"name": name, "smiles": "", "status": "error", "error": str(e)})
        return results

    # ── step 3: structure images ────────────────────────────────

    def _generate_structures(self, resolved: list[dict], style: str) -> list[dict]:
        from src.structure_gen.generator import StructureGenerator

        gen = StructureGenerator()
        results = []
        struct_dir = self.output_dir / "structures"
        struct_dir.mkdir(parents=True, exist_ok=True)

        for r in resolved:
            if r["status"] != "ok" or not r["smiles"]:
                results.append({**r, "struct_path": "", "status": "skip"})
                continue
            try:
                path, _ = gen.generate_from_smiles(
                    r["smiles"],
                    style="ACS_1996",
                    output_path=struct_dir / f"{_safe_name(r['name'])}.png",
                    width=800,
                    height=533,
                )
                results.append({**r, "struct_path": str(path), "status": "ok"})
            except Exception as e:
                logger.warning("structure gen failed for '%s': %s", r["name"], e)
                results.append({**r, "struct_path": "", "status": "error", "error": str(e)})
        return results

    # ── step 4: mindmap (LLM tree) ──────────────────────────────

    def _build_mindmap_from_parsed(self, parsed: dict, resolved: list[dict]) -> str:
        """Build a mindmap using the hierarchical structure from the LLM."""
        from src.mindmap.layout import MindMapLayout, Node

        # Index resolved compounds by name
        smi_map = {r["name"]: r["smiles"] for r in resolved if r["status"] == "ok"}

        # Build node tree from parsed compounds
        nodes: dict[str, Node] = {}
        root = Node(id="root", label=parsed.get("title", "Organic Compounds"))

        for c in parsed.get("compounds", []):
            name = c["name"]
            parent_name = c.get("parent")
            node = Node(id=_safe_name(name), label=name, smiles=smi_map.get(name))
            nodes[name] = node

            if parent_name and parent_name in nodes:
                nodes[parent_name].add_child(node)
            else:
                root.add_child(node)

        layout = MindMapLayout(node_width=200, node_height=150, padding=60)
        img = layout.render(root)

        mindmap_dir = self.output_dir / "mindmaps"
        mindmap_dir.mkdir(parents=True, exist_ok=True)
        path = str(mindmap_dir / "pipeline_mindmap.png")
        cv2.imwrite(path, img)
        return path

    # ── step 5: scene generation ────────────────────────────────

    def _generate_scene(self, prompt: str, style: str, width: int, height: int) -> str:
        from src.scene_gen.generator import SceneGenerator

        gen = SceneGenerator(provider=self._image_provider)
        img = gen.generate(prompt=prompt, width=width, height=height)

        scene_dir = self.output_dir / "scenes"
        scene_dir.mkdir(parents=True, exist_ok=True)
        path = str(scene_dir / "pipeline_scene.png")
        img.save(path)
        return path

    # ── step 6: composite ───────────────────────────────────────

    def _composite(self, scene_path: str, structures: list[dict], width: int, height: int) -> str:
        from src.compositor.basic import load_image, resize_with_alpha
        from src.compositor.lighting import match_and_blend

        bg = load_image(scene_path)
        if bg.shape[:2] != (height, width):
            bg = cv2.resize(bg, (width, height), interpolation=cv2.INTER_LANCZOS4)

        valid = [s for s in structures if s.get("struct_path") and Path(s["struct_path"]).exists()]
        n = len(valid)

        if n == 0:
            final_dir = self.output_dir / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            fp = str(final_dir / "pipeline_final.png")
            cv2.imwrite(fp, bg)
            return fp

        margin = 120
        spacing = (width - 2 * margin) // max(n, 1)
        struct_w = min(220, spacing - 40)
        struct_h = int(struct_w * 0.65)

        for i, s in enumerate(valid):
            overlay = load_image(s["struct_path"])
            if overlay.shape[2] == 3:
                overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

            x = margin + i * spacing + (spacing - struct_w) // 2
            y = height // 2 - struct_h // 2
            overlay_rs = resize_with_alpha(overlay, struct_w, struct_h)
            bg = match_and_blend(
                bg, overlay_rs, (x, y),
                color_match=False, texture_blend=0, shadow=True, feather=0,
            )

        final_dir = self.output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        fp = str(final_dir / "pipeline_final.png")
        cv2.imwrite(fp, bg)
        return fp


def _parse_llm_json(text: str) -> Optional[dict]:
    """Parse JSON from LLM output, handling markdown fences and extra text."""
    # Try to extract from ```json ... ``` fence
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if m:
        text = m.group(1)
    # Find first { ... } block
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def _safe_name(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_一-鿿]', '_', name)[:40]
