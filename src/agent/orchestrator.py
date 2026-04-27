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

    def _default_plan(self, user_input: str) -> dict:
        """Fallback plan when no LLM is available.

        Extracts compound names heuristically and runs the full pipeline.
        """
        import re

        # Heuristic compound detection (look for Chinese/English compound names)
        compounds: list[str] = []
        # Match Chinese compound names (common patterns)
        chinese_matches = re.findall(r'[一-鿿]{1,6}(?:酸|醇|酮|醛|酯|胺|苯|酚|烯|炔|烷|醚|糖|苷|碱)', user_input)
        compounds.extend(chinese_matches)
        # Match English compound patterns
        english_matches = re.findall(r'\b[A-Z][a-z]+(?:ic acid|ol|one|al|ate|ene|ane|yl|ine|ide)\b', user_input)
        compounds.extend(english_matches)

        if not compounds:
            compounds = ["phenol", "benzoic acid"]

        steps: list[dict] = []
        for i, name in enumerate(compounds[:5]):
            steps.append({"tool": "resolve_compound", "params": {"query": name}})

        # After all resolves, generate structures (will be filled post-resolve)
        for i, name in enumerate(compounds[:5]):
            steps.append({
                "tool": "generate_structure",
                "params": {"smiles": f"<from resolve_compound #{i}>", "style": "ACS_1996"},
            })

        # Build mindmap (simplified tree)
        tree = {
            "label": "Organic Compounds",
            "children": [{"label": c} for c in compounds[:5]],
        }
        steps.append({
            "tool": "build_mindmap",
            "params": {"tree_json": json.dumps(tree, ensure_ascii=False), "output_path": "outputs/mindmap.png"},
        })

        steps.append({
            "tool": "generate_scene",
            "params": {"prompt": "academic chemistry mindmap", "style": "academic", "output_path": "outputs/scene.png"},
        })

        steps.append({
            "tool": "detect_surface",
            "params": {"scene_path": "outputs/scene.png", "structure_count": len(compounds[:5])},
        })

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

        for step in plan.get("steps", []):
            tool_name = step["tool"]
            params = dict(step.get("params", {}))

            # Fill in output paths from output_dir
            for key in list(params.keys()):
                if key.endswith("_path") and isinstance(params[key], str) and not params[key].startswith("/"):
                    params[key] = str(output_dir / params[key])

            try:
                tool = self.registry.get(tool_name)
                result = tool(**params)
                results[tool_name] = result
                logger.info("Executed %s → %s", tool_name, result.get("status", "?"))
            except Exception as exc:
                results[tool_name] = {"status": "error", "error": str(exc)}
                logger.error("Tool %s failed: %s", tool_name, exc)

        return results
