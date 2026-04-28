"""Agent orchestrator for chem-mindmap.

Drives the LLM → tool-call → execute → feedback loop to produce a
complete mindmap from a user's natural-language description.

Supports:
- Anthropic Claude API (native tool-use)
- OpenAI API (function calling)
- Default / mock pipeline (no API key required)
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from .tools import ToolRegistry, register_all_tools
from .prompts import SYSTEM_PROMPT, build_user_prompt, build_tools_description

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 12


class AgentOrchestrator:
    """LLM-driven orchestrator that iterates tool calls until completion.

    Usage:
        orch = AgentOrchestrator()
        result = orch.run("生成关于醇类氧化反应的思维导图")
        print(result["final_image"])
    """

    def __init__(self, llm_provider: str = "claude", api_key: Optional[str] = None):
        self.registry = ToolRegistry()
        register_all_tools(self.registry)
        self.llm_provider = llm_provider
        self._api_key = api_key

    # ── public API ──────────────────────────────────────────────

    def run(
        self,
        user_input: str,
        output_dir: Optional[str] = None,
        max_iterations: int = MAX_ITERATIONS,
    ) -> dict:
        """Execute the full generation workflow.

        Args:
            user_input: Natural-language description (e.g. "生成关于苯酚及其衍生物的思维导图")
            output_dir: Output directory override.
            max_iterations: Safety limit for the LLM ↔ tool loop.

        Returns:
            {
                "final_image": "outputs/final/result.png",
                "workflow": [...],
                "results": {...},
                "iterations": N,
            }
        """
        out_dir = Path(output_dir) if output_dir else Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)

        tools_desc = build_tools_description(self.registry.list_tools())
        system = SYSTEM_PROMPT.format(tools_description=tools_desc)

        messages = [
            {"role": "user", "content": build_user_prompt(user_input)},
        ]

        workflow: list[dict] = []

        for iteration in range(1, max_iterations + 1):
            logger.info("=== Iteration %d ===", iteration)

            response = self._call_llm(system, messages, self.registry)

            if response.get("status") == "error":
                return {
                    "final_image": "",
                    "workflow": workflow,
                    "results": {},
                    "iterations": iteration,
                    "error": response.get("error", "LLM call failed"),
                }

            # Check for termination: LLM returned text without tool calls
            if response.get("stop_reason") == "end_turn" and not response.get("tool_calls"):
                logger.info("LLM finished — no more tool calls")
                break

            tool_calls = response.get("tool_calls", [])
            if not tool_calls:
                # Mock / default path: produce a plan and execute it
                plan = response.get("plan")
                if plan == "default" or plan is None:
                    plan = self._default_plan(user_input)
                results = self._execute_plan(plan, out_dir)
                return {
                    "final_image": results.get("composite_final", {}).get("path", ""),
                    "workflow": plan.get("steps", []),
                    "results": results,
                    "iterations": iteration,
                }

            # Execute each tool call, collect results
            tool_results = []
            for tc in tool_calls:
                tool_name = tc["name"]
                tool_input = tc.get("input", {})
                tool_id = tc.get("id", tool_name)

                logger.info("Calling tool: %s(%s)", tool_name, tool_input)

                try:
                    tool = self.registry.get(tool_name)
                    result = tool(**tool_input)
                except KeyError:
                    result = {"status": "error", "error": f"Unknown tool: {tool_name}"}
                except Exception as exc:
                    result = {"status": "error", "error": str(exc)}

                tool_results.append({
                    "tool_use_id": tool_id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

                workflow.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "result": result,
                })

            # Feed tool results back to LLM as user message (Anthropic format)
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": response.get("text", "")}] +
                           [{"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc.get("input", {})}
                            for tc in tool_calls],
            })
            messages.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tr["tool_use_id"], "content": tr["content"]}
                            for tr in tool_results],
            })

            # Check if we have the final composite result
            if any(tc["name"] == "composite_final" for tc in tool_calls):
                final = next((tr for tr in tool_results
                              if json.loads(tr["content"]).get("path")), None)
                if final:
                    path = json.loads(final["content"])["path"]
                    logger.info("Final image: %s", path)
                    return {
                        "final_image": path,
                        "workflow": workflow,
                        "results": {w["tool"]: w["result"] for w in workflow},
                        "iterations": iteration,
                    }

        # Max iterations reached
        return {
            "final_image": "",
            "workflow": workflow,
            "results": {w["tool"]: w["result"] for w in workflow},
            "iterations": max_iterations,
            "warning": "Max iterations reached",
        }

    # ── LLM backends ────────────────────────────────────────────

    def _call_llm(self, system: str, messages: list, registry: ToolRegistry) -> dict:
        """Dispatch to the configured LLM backend."""
        if self.llm_provider == "claude":
            return self._call_claude(system, messages, registry)
        elif self.llm_provider == "openai":
            return self._call_openai(system, messages, registry)
        else:
            return {"plan": "default"}

    def _call_claude(self, system: str, messages: list, registry: ToolRegistry) -> dict:
        """Call Anthropic Claude API with tool-use support.

        API reference: https://docs.anthropic.com/en/docs/build-with-claude/tool-use

        Returns:
            {"tool_calls": [...], "text": "...", "stop_reason": "end_turn" | "tool_use"}
            OR {"plan": "default"} when no API key is available.
        """
        try:
            import anthropic
        except ImportError:
            logger.warning("anthropic package not installed; using default pipeline")
            return {"plan": "default"}

        api_key = self._api_key
        if not api_key:
            logger.warning("No Anthropic API key; using default pipeline")
            return {"plan": "default"}

        client = anthropic.Anthropic(api_key=api_key)

        tools = registry.to_anthropic_tools()

        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=system,
            messages=messages,
            tools=tools,
        )

        tool_calls = []
        text_parts = []

        for block in resp.content:
            if block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
            elif block.type == "text":
                text_parts.append(block.text)

        return {
            "tool_calls": tool_calls,
            "text": "\n".join(text_parts),
            "stop_reason": resp.stop_reason,
        }

    def _call_openai(self, system: str, messages: list, registry: ToolRegistry) -> dict:
        """Call OpenAI API with function-calling support."""
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai package not installed; using default pipeline")
            return {"plan": "default"}

        api_key = self._api_key
        if not api_key:
            logger.warning("No OpenAI API key; using default pipeline")
            return {"plan": "default"}

        client = OpenAI(api_key=api_key)
        tools = registry.to_openai_tools()

        msg_list = [{"role": "system", "content": system}] + messages

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=msg_list,
            tools=tools,
            tool_choice="auto",
            max_tokens=4096,
        )

        choice = resp.choices[0]
        msg = choice.message

        tool_calls = []
        text = msg.content or ""

        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": args,
                })

        return {
            "tool_calls": tool_calls,
            "text": text,
            "stop_reason": "tool_use" if tool_calls else "end_turn",
        }

    # ── default / mock execution ────────────────────────────────

    _EXTRACT_PROMPT = """Extract all organic chemistry compounds from the user's description below.
Return ONLY a valid JSON object (no markdown fences, no extra text) with this exact structure:

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
- Provide accurate canonical SMILES for every compound.
- If the user mentions a reaction, include both reactants and products.
- If the user mentions a compound class (e.g. "alcohols"), expand to 2-3 representative examples.
- Output ONLY the JSON object — no markdown fences, no extra text.

User description:
{user_input}

JSON:"""

    def _llm_extract(self, user_input: str) -> Optional[dict]:
        """Call LLM to extract compounds into structured JSON.

        Returns parsed dict with 'title' and 'compounds' keys, or None on failure.
        Each compound has 'name', 'smiles', 'parent'.
        Uses the configured talk provider (api_config.talk_*).
        """
        from src.config import api_config

        prompt = self._EXTRACT_PROMPT.format(user_input=user_input)
        api_key = self._api_key or api_config.talk_key
        if not api_key:
            logger.info("No API key — cannot extract via LLM")
            return None

        # Always use the configured talk provider (not anthropic)
        try:
            return self._call_openai_text(prompt, api_key)
        except Exception as e:
            logger.warning("LLM extraction failed: %s", e)
            return None

    def _call_openai_text(self, prompt: str, api_key: str) -> Optional[dict]:
        """Single-turn OpenAI-compatible text completion → parsed JSON."""
        import requests
        from src.config import api_config

        url = api_config.talk_url or api_config.talk_base_url + "/v1/chat/completions"
        model = api_config.talk_model("talk") or "gpt-4o"
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
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
        return _parse_json_response(text)

    def _default_plan(self, user_input: str) -> dict:
        """Build an execution plan using LLM compound extraction.

        Calls LLM to parse the user input into structured JSON, then builds
        tool-call steps from the extracted compounds. When LLM is unavailable,
        falls back to a minimal hardcoded list.
        """
        extraction = self._llm_extract(user_input)

        if extraction:
            compounds = extraction.get("compounds", [])
            title = extraction.get("title", "Organic Compounds")
            logger.info("LLM extracted %d compounds: %s", len(compounds),
                        [c.get("name") for c in compounds])
        else:
            logger.info("LLM extraction unavailable — using default compounds")
            compounds = [
                {"name": "phenol", "smiles": "c1ccccc1O", "parent": None},
                {"name": "benzoic acid", "smiles": "O=C(O)c1ccccc1", "parent": None},
            ]
            title = "Organic Compounds"

        n = min(len(compounds), 5)
        compounds = compounds[:n]

        steps: list[dict] = []

        # Step 1: resolve + generate structures
        # When LLM provides SMILES, use directly; otherwise resolve first.
        resolve_idx = 0
        for c in compounds:
            name = c.get("name", str(c)) if isinstance(c, dict) else str(c)
            smiles = c.get("smiles", "") if isinstance(c, dict) else ""

            if smiles:
                # LLM provided SMILES — use directly (skip resolve)
                steps.append({
                    "tool": "generate_structure",
                    "params": {"smiles": smiles, "style": "ACS_1996"},
                })
            else:
                # Need to resolve name → SMILES via PubChem
                steps.append({"tool": "resolve_compound", "params": {"query": name}})
                steps.append({
                    "tool": "generate_structure",
                    "params": {"smiles": f"<from resolve_compound #{resolve_idx}>", "style": "ACS_1996"},
                })
                resolve_idx += 1

        # Step 2: build mindmap tree
        tree = {
            "label": title,
            "children": [
                {
                    "label": c.get("name", str(c)) if isinstance(c, dict) else str(c),
                    "smiles": c.get("smiles", "") if isinstance(c, dict) else "",
                }
                for c in compounds
            ],
        }
        steps.append({
            "tool": "build_mindmap",
            "params": {"tree_json": json.dumps(tree, ensure_ascii=False), "output_path": "outputs/mindmap.png"},
        })

        # Step 3: generate scene
        steps.append({
            "tool": "generate_scene",
            "params": {"prompt": f"academic chemistry mindmap: {title}", "style": "academic", "output_path": "outputs/scene.png"},
        })

        # Step 4: detect surfaces for placement
        steps.append({
            "tool": "detect_surface",
            "params": {"scene_path": "outputs/scene.png", "structure_count": n},
        })

        # Step 5: composite final
        steps.append({
            "tool": "composite_final",
            "params": {
                "scene_path": "outputs/scene.png",
                "structures_info": json.dumps([], ensure_ascii=False),
                "output_path": "outputs/final/result.png",
            },
        })

        return {"steps": steps}

    def _execute_plan(self, plan: dict, output_dir: Path) -> dict:
        """Execute a static tool-call plan step-by-step."""
        results: dict[str, dict] = {}
        resolved_smiles: list[str] = []

        def _fix_path(key: str, value: str) -> str:
            """Prevent path doubling: strip common output prefixes before joining."""
            if value.startswith("/"):
                return value
            # Strip prefixes like "outputs/" or "outputs\\" to avoid doubling
            for prefix in ("outputs/", "outputs\\"):
                if value.startswith(prefix):
                    value = value[len(prefix):]
                    break
            return str(output_dir / value)

        for step in plan.get("steps", []):
            tool_name = step["tool"]
            params = dict(step.get("params", {}))

            # Fill in output paths, avoiding doubling
            for key in list(params.keys()):
                if key.endswith("_path") and isinstance(params[key], str):
                    params[key] = _fix_path(key, params[key])

            # Resolve SMILES placeholders from prior resolve_compound results
            if "smiles" in params and isinstance(params["smiles"], str):
                m = re.match(r'<from resolve_compound #(\d+)>', params["smiles"])
                if m:
                    idx = int(m.group(1))
                    if idx < len(resolved_smiles):
                        params["smiles"] = resolved_smiles[idx]
                    else:
                        logger.warning("Cannot resolve placeholder %s — index %d out of range", params["smiles"], idx)
                        results[tool_name] = {"status": "error", "error": f"Unresolved SMILES placeholder: {params['smiles']}"}
                        continue

            try:
                tool = self.registry.get(tool_name)
                result = tool(**params)
                results[tool_name] = result
                logger.info("Executed %s → %s", tool_name, result.get("status", "?"))

                # Collect SMILES from resolve_compound for later steps
                if tool_name == "resolve_compound" and result.get("status") == "ok":
                    resolved_smiles.append(result.get("smiles", ""))
            except Exception as exc:
                results[tool_name] = {"status": "error", "error": str(exc)}
                logger.error("Tool %s failed: %s", tool_name, exc)

        return results


def _parse_json_response(text: str) -> Optional[dict]:
    """Parse JSON from LLM output, handling markdown fences and stray text."""
    # Strip ```json ... ``` fences
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
