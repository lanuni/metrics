import json
import re
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from .config import AgentRuntimeConfig
from .prompts import (
    CANONICALIZATION,
    DOMAIN_CONTEXT,
    ORCHESTRATOR_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    SPECIALIST_DESCRIPTIONS,
)
from .tools import build_openai_tool_name_map, get_openai_tool_schemas


class AgentState(TypedDict, total=False):
    query: str
    selected_specialists: List[str]
    tool_calls: List[Dict[str, Dict[str, object]]]
    llm_summary: str
    final_response: str
    entities: Dict[str, object]
    reasoning_steps: List[str]


class OpenAILangGraphITSupportAgent:
    """LLM-driven IT support orchestrator using OpenAI tool calling inside a LangGraph workflow."""

    def __init__(self, config: AgentRuntimeConfig | None = None) -> None:
        self.config = config or AgentRuntimeConfig.from_env()
        if not self.config.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Configure it in .env, .envrc, or your shell environment."
            )

        self.client = OpenAI(api_key=self.config.openai_api_key)
        self.openai_to_internal_tool_name = build_openai_tool_name_map()
        self.graph = self._build_graph()

    def act(self, user_query: str) -> List[Dict[str, Dict[str, object]]]:
        return self.run(user_query)["tool_calls"]

    def run(self, user_query: str) -> Dict[str, Any]:
        result = self.graph.invoke({"query": user_query})
        return {
            "query": user_query,
            "normalized_query": user_query.strip().lower(),
            "selected_specialists": result.get("selected_specialists", []),
            "entities": result.get("entities", {}),
            "reasoning_steps": result.get("reasoning_steps", []),
            "tool_calls": result.get("tool_calls", []),
            "final_response": result.get("final_response", "No final response generated."),
        }

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("route_specialists", self._route_specialists)
        graph.add_node("plan_with_llm", self._plan_with_llm)
        graph.add_node("finalize_response", self._finalize_response)
        graph.add_edge(START, "route_specialists")
        graph.add_edge("route_specialists", "plan_with_llm")
        graph.add_edge("plan_with_llm", "finalize_response")
        graph.add_edge("finalize_response", END)
        return graph.compile()

    def _route_specialists(self, state: AgentState) -> AgentState:
        """LLM-based routing. The chosen specialists scope which tools the
        orchestrator can see, so routing is consequential (and measurable),
        not decorative. Falls back to keyword heuristics if the LLM call fails."""
        valid_specialists = list(SPECIALIST_DESCRIPTIONS.keys())
        specialist_context = "\n".join(
            f"- {name}: {SPECIALIST_DESCRIPTIONS[name]}" for name in valid_specialists
        )
        router_prompt = ROUTER_SYSTEM_PROMPT.format(specialists=specialist_context)

        reasoning_step = ""
        selected_specialists: List[str] = []
        try:
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                temperature=self.config.openai_temperature,
                messages=[
                    {"role": "system", "content": router_prompt},
                    {"role": "user", "content": state["query"]},
                ],
            )
            raw = (response.choices[0].message.content or "").strip()
            parsed = self._parse_router_output(raw, valid_specialists)
            selected_specialists = parsed
            reasoning_step = "LLM router selected: " + (", ".join(parsed) if parsed else "none")
        except Exception as exc:  # pragma: no cover - network/runtime guard
            selected_specialists = self._keyword_route(state["query"])
            reasoning_step = (
                f"LLM router failed ({type(exc).__name__}); keyword fallback selected: "
                + ", ".join(selected_specialists)
            )

        return {
            "selected_specialists": selected_specialists,
            "reasoning_steps": [reasoning_step],
        }

    def _parse_router_output(self, raw: str, valid_specialists: List[str]) -> List[str]:
        """Parse the router's JSON array, tolerating extra prose. An empty array
        is a valid answer (conversational / out-of-scope request)."""
        candidate = raw
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return self._keyword_route(raw)

        if not isinstance(parsed, list):
            return self._keyword_route(raw)

        ordered = [name for name in valid_specialists if name in parsed]
        return ordered

    def _keyword_route(self, query: str) -> List[str]:
        lowered = query.lower()
        selected: List[str] = []
        if any(token in lowered for token in ["contrase", "desbloq", "mfa", "identidad", "acceso"]):
            selected.append("identity_specialist")
        if any(token in lowered for token in ["ticket", "incidente", "servicio", "mantenimiento", "escal"]):
            selected.append("operations_specialist")
        if any(token in lowered for token in ["kb", "guia", "guía", "runbook", "documentacion", "documentación"]):
            selected.append("knowledge_specialist")
        return selected

    def _plan_with_llm(self, state: AgentState) -> AgentState:
        selected_specialists = state.get("selected_specialists", [])
        specialist_context = "\n".join(
            f"- {name}: {SPECIALIST_DESCRIPTIONS[name]}" for name in selected_specialists
        ) or "- (none selected; treat as conversational / out of scope)"
        system_prompt = ORCHESTRATOR_SYSTEM_PROMPT.format(
            specialists=specialist_context,
            canonicalization=CANONICALIZATION,
            domain_context=DOMAIN_CONTEXT,
        )

        # Only the routed specialists' tools are exposed. If routing returned no
        # specialist, no tools are offered and the agent must answer conversationally.
        scoped_tools = get_openai_tool_schemas(specialists=selected_specialists)

        request_kwargs: Dict[str, Any] = {
            "model": self.config.openai_model,
            "temperature": self.config.openai_temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": state["query"]},
            ],
        }
        if scoped_tools:
            request_kwargs["tools"] = scoped_tools
            request_kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**request_kwargs)

        message = response.choices[0].message
        tool_calls = self._convert_tool_calls(message.tool_calls or [])
        llm_summary = message.content or self._fallback_summary(tool_calls)
        entities = self._extract_entities_from_tool_calls(tool_calls)
        reasoning_steps = list(state.get("reasoning_steps", []))
        reasoning_steps.append(f"LLM orchestration summary: {llm_summary}")

        return {
            "tool_calls": tool_calls,
            "llm_summary": llm_summary,
            "entities": entities,
            "reasoning_steps": reasoning_steps,
        }

    def _finalize_response(self, state: AgentState) -> AgentState:
        tool_calls = state.get("tool_calls", [])
        if not tool_calls:
            final_response = "No se requiere tool call. La solicitud parece conversacional o fuera del dominio operativo."
        else:
            tool_names = [list(call.keys())[0] for call in tool_calls]
            final_response = "Plan operativo generado con las siguientes herramientas: " + ", ".join(tool_names)

        return {"final_response": final_response}

    def _convert_tool_calls(self, openai_tool_calls) -> List[Dict[str, Dict[str, object]]]:
        converted_calls: List[Dict[str, Dict[str, object]]] = []
        for tool_call in openai_tool_calls:
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            openai_name = tool_call.function.name
            internal_name = self.openai_to_internal_tool_name.get(openai_name, openai_name)
            converted_calls.append({internal_name: arguments})
        return converted_calls

    def _extract_entities_from_tool_calls(
        self, tool_calls: List[Dict[str, Dict[str, object]]]
    ) -> Dict[str, object]:
        aggregated_entities: Dict[str, object] = {}
        for call in tool_calls:
            params = list(call.values())[0]
            for key, value in params.items():
                if key not in aggregated_entities:
                    aggregated_entities[key] = value
        return aggregated_entities

    def _fallback_summary(self, tool_calls: List[Dict[str, Dict[str, object]]]) -> str:
        if not tool_calls:
            return "The request does not require operational tool usage."
        tool_names = [list(call.keys())[0] for call in tool_calls]
        return "Selected tool calls: " + ", ".join(tool_names)