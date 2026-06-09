import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str
    properties: Dict[str, Dict[str, Any]]
    required: List[str]
    # Allowed values per parameter. The agent is expected to emit one of these
    # canonical values (BFCL-style enum constraints, like a real function API).
    enum: Dict[str, List[str]] = field(default_factory=dict)
    # Free-text parameters whose exact value should NOT be matched during
    # evaluation (e.g. an incident description). They only need to be present
    # and non-empty.
    freeform: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Controlled vocabulary (single source of truth)
#
# The agent prompt teaches these mappings so the LLM produces canonical values,
# and the notebook evaluator imports them so matching is alias-tolerant. Keeping
# them here avoids drift between what the agent is told and how it is scored.
# ---------------------------------------------------------------------------

SERVICES = ["email", "vpn", "jira", "sap", "network", "identity"]
SEVERITIES = ["critical", "high", "medium", "low"]
PRIORITIES = ["p1", "p2", "p3", "p4"]
ENVIRONMENTS = ["production", "qa", "staging", "dev"]
CHANNELS = ["email", "sms", "whatsapp", "phone"]
UNLOCK_SYSTEMS = ["identity", "vpn", "email"]
ASSIGNMENT_GROUPS = ["messaging_ops", "network_ops", "identity_ops", "app_ops", "infra_ops"]

# Extended catalog: richer tools with arrays, dates and many parameters.
ACCESS_SYSTEMS = ["sap", "jira", "vpn", "email", "github", "aws", "network", "identity"]
ACCESS_ROLES = ["reader", "contributor", "admin", "auditor"]
DEPARTMENTS = ["engineering", "finance", "hr", "sales", "it", "legal"]
USER_GROUPS = ["vpn", "email", "jira", "sap", "github"]
HARDWARE_ITEMS = ["laptop", "monitor", "keyboard", "dock", "headset", "phone"]
SHIPPING_REGIONS = ["us-east", "us-west", "eu-west", "latam", "apac"]
TICKET_STATUSES = ["open", "in_progress", "on_hold", "resolved", "closed"]
TICKET_TAGS = ["bug", "access", "network", "hardware", "billing"]
DIAGNOSTIC_CHECKS = ["latency", "packet_loss", "dns", "auth", "throughput"]

# Controlled knowledge-base topic catalog. Free-text requests are mapped to the
# nearest catalog entry by the agent; the evaluator fuzzy-matches against it.
KB_TOPICS = [
    "mfa enrollment",
    "sap connectivity",
    "vpn setup",
    "password policy",
    "email troubleshooting",
    "jira access",
    "network outage",
]

# Spanish (and informal) -> canonical value maps.
SERVICE_ALIASES = {
    "correo": "email",
    "mail": "email",
    "e-mail": "email",
    "vpn": "vpn",
    "jira": "jira",
    "sap": "sap",
    "red": "network",
    "network": "network",
    "identidad": "identity",
    "identity": "identity",
}

SEVERITY_ALIASES = {
    "critica": "critical",
    "crítica": "critical",
    "critico": "critical",
    "crítico": "critical",
    "critical": "critical",
    "alta": "high",
    "high": "high",
    "media": "medium",
    "medium": "medium",
    "baja": "low",
    "low": "low",
}

# Escalation priority follows a documented business rule (urgency -> priority).
PRIORITY_ALIASES = {
    "urgente": "p1",
    "critica": "p1",
    "crítica": "p1",
    "p1": "p1",
    "alta": "p2",
    "high": "p2",
    "p2": "p2",
    "media": "p3",
    "medium": "p3",
    "p3": "p3",
    "baja": "p4",
    "low": "p4",
    "p4": "p4",
}

ENVIRONMENT_ALIASES = {
    "produccion": "production",
    "producción": "production",
    "prod": "production",
    "production": "production",
    "qa": "qa",
    "staging": "staging",
    "dev": "dev",
    "desarrollo": "dev",
}

ASSIGNMENT_ALIASES = {
    "correo": "messaging_ops",
    "email": "messaging_ops",
    "mensajeria": "messaging_ops",
    "messaging_ops": "messaging_ops",
    "red": "network_ops",
    "network": "network_ops",
    "network_ops": "network_ops",
    "identidad": "identity_ops",
    "identity": "identity_ops",
    "identity_ops": "identity_ops",
    "aplicaciones": "app_ops",
    "apps": "app_ops",
    "app_ops": "app_ops",
    "infraestructura": "infra_ops",
    "infra": "infra_ops",
    "infra_ops": "infra_ops",
}

DEPARTMENT_ALIASES = {
    "ingenieria": "engineering",
    "ingeniería": "engineering",
    "engineering": "engineering",
    "finanzas": "finance",
    "finance": "finance",
    "rrhh": "hr",
    "recursos humanos": "hr",
    "hr": "hr",
    "ventas": "sales",
    "sales": "sales",
    "ti": "it",
    "sistemas": "it",
    "it": "it",
    "legales": "legal",
    "legal": "legal",
}

ITEM_ALIASES = {
    "laptop": "laptop",
    "notebook": "laptop",
    "portatil": "laptop",
    "portátil": "laptop",
    "monitor": "monitor",
    "pantalla": "monitor",
    "teclado": "keyboard",
    "keyboard": "keyboard",
    "dock": "dock",
    "base": "dock",
    "docking": "dock",
    "auriculares": "headset",
    "headset": "headset",
    "telefono": "phone",
    "teléfono": "phone",
    "celular": "phone",
    "phone": "phone",
}

STATUS_ALIASES = {
    "abierto": "open",
    "open": "open",
    "en progreso": "in_progress",
    "en curso": "in_progress",
    "in_progress": "in_progress",
    "en espera": "on_hold",
    "pausado": "on_hold",
    "on_hold": "on_hold",
    "resuelto": "resolved",
    "resolved": "resolved",
    "cerrado": "closed",
    "closed": "closed",
}

SYSTEM_ALIASES = {
    "correo": "email",
    "mail": "email",
    "email": "email",
    "red": "network",
    "network": "network",
    "identidad": "identity",
    "identity": "identity",
    "github": "github",
    "aws": "aws",
    "sap": "sap",
    "jira": "jira",
    "vpn": "vpn",
}

CHECK_ALIASES = {
    "latencia": "latency",
    "latency": "latency",
    "perdida de paquetes": "packet_loss",
    "pérdida de paquetes": "packet_loss",
    "packet_loss": "packet_loss",
    "dns": "dns",
    "autenticacion": "auth",
    "autenticación": "auth",
    "auth": "auth",
    "throughput": "throughput",
}

# Per-parameter alias tables, consumed by the evaluator to map a predicted value
# onto its canonical form before comparison.
VALUE_ALIASES: Dict[str, Dict[str, str]] = {
    "service": SERVICE_ALIASES,
    "product": SERVICE_ALIASES,
    "severity": SEVERITY_ALIASES,
    "priority": PRIORITY_ALIASES,
    "assignment_group": ASSIGNMENT_ALIASES,
    "environment": ENVIRONMENT_ALIASES,
    "department": DEPARTMENT_ALIASES,
    "item": ITEM_ALIASES,
    "status": STATUS_ALIASES,
    "system": SYSTEM_ALIASES,
    "checks": CHECK_ALIASES,
}


def canonicalize_datetime(value: str) -> str:
    """Normalize a datetime-like string to digits only, so different separators
    (``2026-06-10 22:00`` vs ``2026-06-10T22:00``) compare equal."""
    if not isinstance(value, str):
        return value
    return re.sub(r"[^0-9]", "", value)


# ---------------------------------------------------------------------------
# Specialist ownership: which specialist is allowed to use which tools. The
# router selects specialists and only their tools are exposed to the orchestrator.
# ---------------------------------------------------------------------------

SPECIALIST_TOOLS: Dict[str, List[str]] = {
    "identity_specialist": [
        "it_support.reset_password",
        "it_support.unlock_account",
    ],
    "operations_specialist": [
        "it_support.check_ticket_status",
        "it_support.create_incident",
        "it_support.get_service_status",
        "it_support.escalate_ticket",
        "it_support.schedule_maintenance",
        "it_support.update_ticket",
        "it_support.run_diagnostic",
    ],
    "knowledge_specialist": [
        "it_support.search_kb_article",
    ],
    "provisioning_specialist": [
        "it_support.provision_access",
        "it_support.create_user",
        "it_support.order_hardware",
    ],
}


def tool_owner(tool_name: str) -> Optional[str]:
    """Return the specialist that owns a tool, or None if unknown."""
    for specialist, tools in SPECIALIST_TOOLS.items():
        if tool_name in tools:
            return specialist
    return None


def to_openai_tool_name(internal_name: str) -> str:
    """Convert internal tool names to OpenAI-compatible function names."""
    return internal_name.replace(".", "_")


def build_openai_tool_name_map() -> Dict[str, str]:
    """Map OpenAI-compatible names back to internal tool names."""
    return {
        to_openai_tool_name(schema.name): schema.name
        for schema in get_tool_schemas()
    }


def _all_tool_schemas() -> List[ToolSchema]:
    return [
        ToolSchema(
            name="it_support.reset_password",
            description="Resets a user password and optionally notifies the user.",
            properties={
                "username": {"type": "string"},
                "channel": {"type": "string"},
                "urgent": {"type": "boolean"},
            },
            required=["username"],
            enum={"channel": CHANNELS},
        ),
        ToolSchema(
            name="it_support.unlock_account",
            description="Unlocks a user account in the identity or access platform.",
            properties={
                "username": {"type": "string"},
                "system": {"type": "string"},
            },
            required=["username"],
            enum={"system": UNLOCK_SYSTEMS},
        ),
        ToolSchema(
            name="it_support.check_ticket_status",
            description="Returns the current status of an incident ticket.",
            properties={
                "ticket_id": {"type": "string"},
            },
            required=["ticket_id"],
        ),
        ToolSchema(
            name="it_support.create_incident",
            description="Creates a new incident ticket with service and severity.",
            properties={
                "service": {"type": "string"},
                "severity": {"type": "string"},
                "description": {"type": "string"},
                "environment": {"type": "string"},
            },
            required=["service", "severity", "description"],
            enum={
                "service": SERVICES,
                "severity": SEVERITIES,
                "environment": ENVIRONMENTS,
            },
            freeform=["description"],
        ),
        ToolSchema(
            name="it_support.get_service_status",
            description="Checks current health for an IT service.",
            properties={
                "service": {"type": "string"},
                "region": {"type": "string"},
            },
            required=["service"],
            enum={"service": SERVICES},
        ),
        ToolSchema(
            name="it_support.search_kb_article",
            description="Searches the internal knowledge base for runbooks or help articles.",
            properties={
                "topic": {"type": "string"},
                "product": {"type": "string"},
            },
            required=["topic"],
            enum={"product": SERVICES},
        ),
        ToolSchema(
            name="it_support.escalate_ticket",
            description="Escalates a ticket to a specialized assignment group with a target priority.",
            properties={
                "ticket_id": {"type": "string"},
                "assignment_group": {"type": "string"},
                "priority": {"type": "string"},
            },
            required=["ticket_id", "assignment_group", "priority"],
            enum={
                "assignment_group": ASSIGNMENT_GROUPS,
                "priority": PRIORITIES,
            },
        ),
        ToolSchema(
            name="it_support.schedule_maintenance",
            description="Schedules a maintenance window for a service.",
            properties={
                "service": {"type": "string"},
                "start_time": {"type": "string"},
                "duration_minutes": {"type": "integer"},
                "reason": {"type": "string"},
            },
            required=["service", "start_time", "duration_minutes", "reason"],
            enum={"service": SERVICES},
            freeform=["reason"],
        ),
        ToolSchema(
            name="it_support.update_ticket",
            description="Updates an existing ticket: status, assignee, free-text comment and tags.",
            properties={
                "ticket_id": {"type": "string"},
                "status": {"type": "string"},
                "assignee": {"type": "string"},
                "comment": {"type": "string"},
                "tags": {"type": "array"},
            },
            required=["ticket_id"],
            enum={"status": TICKET_STATUSES, "tags": TICKET_TAGS},
            freeform=["comment"],
        ),
        ToolSchema(
            name="it_support.run_diagnostic",
            description="Runs health diagnostics on a service for one or more requested checks.",
            properties={
                "service": {"type": "string"},
                "checks": {"type": "array"},
                "region": {"type": "string"},
            },
            required=["service", "checks"],
            enum={"service": SERVICES, "checks": DIAGNOSTIC_CHECKS},
        ),
        ToolSchema(
            name="it_support.provision_access",
            description="Grants a user one or more roles on a system, with a justification and optional approver and duration.",
            properties={
                "username": {"type": "string"},
                "system": {"type": "string"},
                "roles": {"type": "array"},
                "duration_days": {"type": "integer"},
                "justification": {"type": "string"},
                "approver": {"type": "string"},
            },
            required=["username", "system", "roles", "justification"],
            enum={"system": ACCESS_SYSTEMS, "roles": ACCESS_ROLES},
            freeform=["justification"],
        ),
        ToolSchema(
            name="it_support.create_user",
            description="Creates a new user account in a department and optionally adds access groups.",
            properties={
                "full_name": {"type": "string"},
                "username": {"type": "string"},
                "department": {"type": "string"},
                "manager": {"type": "string"},
                "start_date": {"type": "string"},
                "groups": {"type": "array"},
            },
            required=["full_name", "username", "department"],
            enum={"department": DEPARTMENTS, "groups": USER_GROUPS},
        ),
        ToolSchema(
            name="it_support.order_hardware",
            description="Places a hardware order for an item, quantity and shipping region.",
            properties={
                "item": {"type": "string"},
                "quantity": {"type": "integer"},
                "cost_center": {"type": "string"},
                "shipping_region": {"type": "string"},
                "priority": {"type": "string"},
            },
            required=["item", "quantity"],
            enum={"item": HARDWARE_ITEMS, "shipping_region": SHIPPING_REGIONS, "priority": PRIORITIES},
        ),
    ]


def get_tool_schemas(specialists: Optional[List[str]] = None) -> List[ToolSchema]:
    """Return tool schemas, optionally restricted to the tools owned by the
    given specialists. With ``specialists=None`` the full catalog is returned."""
    schemas = _all_tool_schemas()
    if specialists is None:
        return schemas

    allowed: List[str] = []
    for specialist in specialists:
        allowed.extend(SPECIALIST_TOOLS.get(specialist, []))
    allowed_set = set(allowed)
    return [schema for schema in schemas if schema.name in allowed_set]


def get_openai_tool_schemas(specialists: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    openai_tools: List[Dict[str, Any]] = []
    for schema in get_tool_schemas(specialists):
        properties: Dict[str, Any] = {}
        for param_name, param_spec in schema.properties.items():
            param_payload = dict(param_spec)
            enum_values = schema.enum.get(param_name)
            if param_payload.get("type") == "array":
                # Array enums belong on the items, not the array itself.
                items_spec: Dict[str, Any] = {"type": "string"}
                if enum_values:
                    items_spec["enum"] = enum_values
                param_payload["items"] = items_spec
            elif enum_values:
                param_payload["enum"] = enum_values
            properties[param_name] = param_payload

        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": to_openai_tool_name(schema.name),
                    "description": schema.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": schema.required,
                    },
                },
            }
        )
    return openai_tools


def reset_password(username: str, channel: str = "email", urgent: bool = False) -> Dict[str, Any]:
    return {
        "function": "it_support.reset_password",
        "parameters": {"username": username, "channel": channel, "urgent": urgent},
    }


def unlock_account(username: str, system: str = "identity") -> Dict[str, Any]:
    return {
        "function": "it_support.unlock_account",
        "parameters": {"username": username, "system": system},
    }


def check_ticket_status(ticket_id: str) -> Dict[str, Any]:
    return {
        "function": "it_support.check_ticket_status",
        "parameters": {"ticket_id": ticket_id},
    }


def create_incident(
    service: str,
    severity: str,
    description: str,
    environment: str = "production",
) -> Dict[str, Any]:
    return {
        "function": "it_support.create_incident",
        "parameters": {
            "service": service,
            "severity": severity,
            "description": description,
            "environment": environment,
        },
    }


def get_service_status(service: str, region: str = "global") -> Dict[str, Any]:
    return {
        "function": "it_support.get_service_status",
        "parameters": {"service": service, "region": region},
    }


def search_kb_article(topic: str, product: str = "") -> Dict[str, Any]:
    return {
        "function": "it_support.search_kb_article",
        "parameters": {"topic": topic, "product": product},
    }


def escalate_ticket(ticket_id: str, assignment_group: str, priority: str) -> Dict[str, Any]:
    return {
        "function": "it_support.escalate_ticket",
        "parameters": {
            "ticket_id": ticket_id,
            "assignment_group": assignment_group,
            "priority": priority,
        },
    }


def schedule_maintenance(
    service: str,
    start_time: str,
    duration_minutes: int,
    reason: str,
) -> Dict[str, Any]:
    return {
        "function": "it_support.schedule_maintenance",
        "parameters": {
            "service": service,
            "start_time": start_time,
            "duration_minutes": duration_minutes,
            "reason": reason,
        },
    }


def update_ticket(
    ticket_id: str,
    status: str = "",
    assignee: str = "",
    comment: str = "",
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "function": "it_support.update_ticket",
        "parameters": {
            "ticket_id": ticket_id,
            "status": status,
            "assignee": assignee,
            "comment": comment,
            "tags": tags or [],
        },
    }


def run_diagnostic(
    service: str,
    checks: List[str],
    region: str = "global",
) -> Dict[str, Any]:
    return {
        "function": "it_support.run_diagnostic",
        "parameters": {"service": service, "checks": checks, "region": region},
    }


def provision_access(
    username: str,
    system: str,
    roles: List[str],
    justification: str,
    duration_days: int = 0,
    approver: str = "",
) -> Dict[str, Any]:
    return {
        "function": "it_support.provision_access",
        "parameters": {
            "username": username,
            "system": system,
            "roles": roles,
            "duration_days": duration_days,
            "justification": justification,
            "approver": approver,
        },
    }


def create_user(
    full_name: str,
    username: str,
    department: str,
    manager: str = "",
    start_date: str = "",
    groups: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "function": "it_support.create_user",
        "parameters": {
            "full_name": full_name,
            "username": username,
            "department": department,
            "manager": manager,
            "start_date": start_date,
            "groups": groups or [],
        },
    }


def order_hardware(
    item: str,
    quantity: int,
    cost_center: str = "",
    shipping_region: str = "",
    priority: str = "",
) -> Dict[str, Any]:
    return {
        "function": "it_support.order_hardware",
        "parameters": {
            "item": item,
            "quantity": quantity,
            "cost_center": cost_center,
            "shipping_region": shipping_region,
            "priority": priority,
        },
    }
