"""
Metrics Agent -- agent ReAct (LangGraph) + Gemini 2.5 Flash, connecté au
serveur MCP maison (metrics_mcp_server.py) exposant des tools métier sur
data/cloudwatch_metrics.db.

L'agent utilise ces tools pour interroger la table `cloudwatch_metrics`
avant de rendre son verdict structuré.

Prérequis:
    export GOOGLE_API_KEY="..."          # clé Google AI Studio
    pip install -r ../requirements.txt

Usage:
    python agent.py --instance ec2_cpu_utilization_24ae8d --metric CPUUtilization
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

sys.path.insert(0, os.path.dirname(__file__))
from prompt import METRICS_AGENT_SYSTEM_PROMPT

MCP_CONFIG = {
    "metrics-agent-tools": {
        "command": "python",
        "args": [
            os.path.join(os.path.dirname(__file__), "metrics_mcp_server.py"),
        ],
        "transport": "stdio",
    }
}


def normalize_content(content) -> str:
    """
    Les versions récentes de langchain-google-genai peuvent renvoyer le
    contenu du message sous forme de liste de blocs (ex: [{"type": "text",
    "text": "..."}]) plutôt qu'une simple string -- on normalise ici.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def extract_json(content) -> dict:
    """Le LLM peut entourer le JSON de ```json ... ``` ou de texte -- on isole l'objet."""
    text = normalize_content(content)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Aucun JSON trouvé dans la réponse:\n{text}")
    return json.loads(match.group(0))


def _extract_retry_delay(message: str) -> int:
    """Cherche 'retryDelay': '37s' dans le message d'erreur brut de l'API."""
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+)s", message)
    return int(match.group(1)) if match else None


def _classify_quota_error(message: str) -> str:
    """
    Distingue RPM (per-minute, se résout en attendant) de RPD (per-day, ne se
    résout PAS en attendant quelques secondes -- il faut attendre le reset
    quotidien ou passer à un tier payant). On se base sur les mots-clés
    présents dans le message d'erreur Google (souvent 'PerDay' vs 'PerMinute'
    dans le nom du quota violé).
    """
    lowered = message.lower()
    if "perday" in lowered or "per_day" in lowered or "daily" in lowered:
        return "RPD (quota journalier)"
    if "perminute" in lowered or "per_minute" in lowered:
        return "RPM (quota par minute)"
    return "type de quota inconnu -- voir message brut ci-dessous"


async def _ainvoke_with_retry(agent, payload, max_retries: int = 6):
    """
    Réessaie l'appel agent en cas de 429 RESOURCE_EXHAUSTED. langchain-google-genai
    enrobe l'erreur Google dans sa propre exception (pas google.api_core direct),
    donc on détecte le cas par le contenu du message plutôt que par le type.
    """
    for attempt in range(max_retries):
        try:
            return await agent.ainvoke(payload)
        except Exception as exc:
            msg = str(exc)
            is_quota_error = "RESOURCE_EXHAUSTED" in msg or "429" in msg
            if not is_quota_error or attempt == max_retries - 1:
                raise
            wait_s = _extract_retry_delay(msg) or min(2 ** attempt * 5, 60)
            wait_s += 2  # petite marge de sécurité
            quota_type = _classify_quota_error(msg)
            print(
                f"  [retry] quota atteint ({quota_type}), nouvelle tentative dans {wait_s}s "
                f"(essai {attempt + 1}/{max_retries})",
                file=sys.stderr,
            )
            if attempt == 0:
                # Affiche le message brut une seule fois (pas à chaque retry, pour
                # ne pas noyer la sortie) -- utile pour diagnostiquer précisément
                # quel quota est touché si la classification ci-dessus échoue.
                print(f"  [debug] message brut: {msg[:500]}", file=sys.stderr)
            time.sleep(wait_s)
    raise RuntimeError("Nombre maximal de tentatives dépassé")


AGENT_DEBUG = os.environ.get("AGENT_DEBUG", "0") == "1"


def _print_tool_call_trace(messages: list) -> None:
    """
    Affiche, dans l'ordre, chaque tool appelé par l'agent pendant la boucle
    ReAct (nom + arguments choisis par le LLM) ainsi qu'un aperçu du résultat
    renvoyé par le tool -- utile pour vérifier que describe_metrics_schema
    est bien appelé avant get_metric_points, que get_baseline_stats est bien
    appelé avant le verdict final, ou pour voir combien de tours l'agent a
    faits avant de conclure. Activé via AGENT_DEBUG=1.
    """
    if not AGENT_DEBUG:
        return
    print("  --- trace des tool calls ---", file=sys.stderr)
    step = 0
    for msg in messages:
        # Un AIMessage peut porter une ou plusieurs demandes de tool call.
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                step += 1
                print(f"  [{step}] APPEL {tc['name']}({tc['args']})", file=sys.stderr)
        # Un ToolMessage porte le résultat renvoyé par le serveur MCP.
        if getattr(msg, "type", None) == "tool":
            preview = normalize_content(msg.content)[:300]
            print(f"      -> résultat: {preview}", file=sys.stderr)
    print("  --- fin de trace ---", file=sys.stderr)


async def run_metrics_agent(
    instance_id: str, metric_name: str, end_timestamp: str | None = None
) -> dict:
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY non défini. export GOOGLE_API_KEY=...")

    model_name = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    client = MultiServerMCPClient(MCP_CONFIG)
    tools = await client.get_tools()

    agent = create_react_agent(llm, tools, prompt=METRICS_AGENT_SYSTEM_PROMPT)

    if end_timestamp:
        user_message = (
            f"Analyse l'instance `{instance_id}` pour la métrique `{metric_name}`. "
            f"Utilise d'abord `describe_metrics_schema` pour vérifier que cette métrique "
            f"existe bien pour cette instance, puis `get_metric_points` (end_timestamp="
            f"'{end_timestamp}', limit=30) pour récupérer les points se terminant le plus "
            f"près possible de cette date. Utilise ensuite `get_baseline_stats` avec "
            f"before_timestamp égal au timestamp du premier point de cette fenêtre, pour "
            f"obtenir un seuil de référence calculé sur l'historique -- ne l'invente jamais. "
            f"Rends ensuite ton verdict au format JSON demandé."
        )
    else:
        user_message = (
            f"Analyse l'instance `{instance_id}` pour la métrique `{metric_name}`. "
            f"Utilise d'abord `describe_metrics_schema` pour vérifier que cette métrique "
            f"existe bien pour cette instance, puis `get_metric_points` (limit=30) pour "
            f"récupérer les 30 derniers points. Utilise ensuite `get_baseline_stats` avec "
            f"before_timestamp égal au timestamp du premier point de cette fenêtre, pour "
            f"obtenir un seuil de référence calculé sur l'historique -- ne l'invente jamais. "
            f"Rends ensuite ton verdict au format JSON demandé."
        )

    result = await _ainvoke_with_retry(
        agent, {"messages": [{"role": "user", "content": user_message}]}
    )
    _print_tool_call_trace(result["messages"])
    final_text = result["messages"][-1].content

    return extract_json(final_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--metric", default="CPUUtilization")
    parser.add_argument(
        "--end-timestamp", default=None,
        help="Fin de la fenêtre de 30 points à analyser (YYYY-MM-DD HH:MM:SS). "
             "Par défaut: les 30 derniers points de la série."
    )
    args = parser.parse_args()

    output = asyncio.run(
        run_metrics_agent(args.instance, args.metric, args.end_timestamp)
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()