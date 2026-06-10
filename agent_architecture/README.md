# Agente de Soporte IT Para Evaluacion BFCL-Style

Esta carpeta contiene una arquitectura local, reproducible y algo mas realista para evaluar un agente, no solo un LLM aislado.

## Dominio
- Soporte IT interno para servicios como email, VPN, Jira, SAP, red e identidad.
- El agente opera sobre solicitudes frecuentes de service desk: credenciales, tickets, incidentes, runbooks y mantenimientos.

## Herramientas disponibles (13)
Credenciales / identidad:
- `it_support.reset_password`
- `it_support.unlock_account`

Operaciones:
- `it_support.check_ticket_status`
- `it_support.create_incident`
- `it_support.get_service_status`
- `it_support.escalate_ticket`
- `it_support.schedule_maintenance`
- `it_support.update_ticket` (status/assignee/comment + `tags` array)
- `it_support.run_diagnostic` (`checks` array)

Conocimiento:
- `it_support.search_kb_article`

Provisioning:
- `it_support.provision_access` (`roles` array, justificacion free-text, aprobador)
- `it_support.create_user` (department, start_date, `groups` array)
- `it_support.order_hardware` (item, quantity, shipping_region)

## Arquitectura del agente
- `llm_agent.py`: agente LLM-driven usando OpenAI tool calling dentro de un flujo LangGraph de tres pasos (routing -> orquestacion con tools acotadas -> respuesta).
- `factory.py`: constructor del agente (LLM-only).
- `config.py`: carga de entorno (`.env`, `.envrc`) y parametros del runtime.
- `prompts.py`: system prompt de orquestacion, prompt del router y vocabulario de canonicalizacion.
- `tools.py`: catalogo de herramientas con **enums** (valores permitidos) y campos `freeform`, **ownership** de herramientas por especialista (`SPECIALIST_TOOLS`), y el **vocabulario controlado** compartido (alias español->canonico, catalogo de topics KB, helpers de normalizacion) que tambien usa el evaluador.
- `bfcl_like_cases.json`: benchmark local (31 casos) con categorias `simple`, `parallel`, `complex` (una herramienta con muchos parametros / arrays), `relevance` e `irrelevance` (con distractores).

### Routing consecuente (no decorativo)
- Un **router LLM** (`ROUTER_SYSTEM_PROMPT`) decide que especialistas (`identity` / `operations` / `knowledge` / `provisioning`) son relevantes; con fallback a heuristica por keywords si la llamada falla.
- En la orquestacion solo se exponen las herramientas de los especialistas elegidos (`get_openai_tool_schemas(specialists=...)`). Un ruteo equivocado deja al agente sin las herramientas necesarias, por lo que el ruteo es medible: el notebook compara los especialistas elegidos contra los *dueños* de las herramientas esperadas.

### Canonicalizacion
- Las herramientas declaran enums (p. ej. `severity ∈ {critical, high, medium, low}`, `priority ∈ {p1..p4}`, `service ∈ {email, vpn, jira, sap, network, identity}`).
- El agente recibe un mapa español->canonico (`correo`→`email`, `critica`→`critical`, `equipo de correo`→`messaging_ops`, `prioridad alta`→`p2`, formato de fecha `YYYY-MM-DD HH:MM`) para emitir argumentos validos. El evaluador es **tolerante a alias** usando el mismo mapa.

## Variables de entorno
- `OPENAI_API_KEY=...`
- `OPENAI_MODEL=gpt-4.1-mini`
- `OPENAI_TEMPERATURE=0`

El backend rule-based ya no existe. Este proyecto evalua exclusivamente un agente impulsado por LLM.

## Uso rapido

```python
from agent_architecture import build_agent

agent = build_agent()
trace = agent.run("Consulta el ticket inc-9001 y escalalo al equipo de correo con prioridad alta")

print(type(agent).__name__)
print(trace["tool_calls"])
print(trace["reasoning_steps"])
```

## Objetivo de evaluacion
El notebook asociado mide exactitud de function calling inspirada en Berkeley Function Calling Leaderboard (BFCL), con chequeo AST-style estricto sobre:
- nombre de funcion
- parametros requeridos y opcionales
- parametros inesperados
- pertenencia al enum cuando aplica
- tipo y valor (con matching tolerante a alias + normalizacion de fechas)
- parametros free-text: presentes y no vacios (sin match exacto)
- matching all-or-nothing en casos con multiples llamadas

Ademas calcula metricas de **routing** (accuracy/precision/recall) comparando los especialistas elegidos por el router contra los dueños de las herramientas esperadas, y muestra trazas del agente: decision de ruteo, herramientas elegidas y respuesta operativa final.

## Integracion LLM
La variante LLM-backed usa OpenAI Chat Completions con function calling y un grafo LangGraph de tres pasos:
- routing de especialistas
- orquestacion LLM con tools
- construccion de respuesta final

Esto permite evaluar un agente ficticio pero realmente impulsado por un LLM, manteniendo el mismo formato estructurado de salida que necesita la metrica BFCL-style.
