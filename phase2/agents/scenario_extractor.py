"""
Scenario Extractor Agent — Layer 1

Core question: What game is this? What are H1/H2/H3?

Responsibilities:
- Read paper_parse.json and extract game structure
- Formalize hypotheses with nature tags
- Write paradigm.yaml
- Iterate with Paper Agent (query) and Solver Builder (negotiate testability)
"""

import logging
import time

import yaml

from tools import file_io as fio
from tools.agent_api_client import AgentClient, make_tool_spec

logger = logging.getLogger(__name__)

AGENT_NAME = "scenario_extractor"

SYSTEM_PROMPT = """You are the Scenario Extractor Agent in a multi-agent research framework.

Your job is to read a structured paper parse and extract:
1. The game structure (players, actions, equilibrium concept)
2. Key parameters and their economic meaning
3. Three hypotheses (H1, H2, H3) that capture the paper's core testable claims

For Scenario B ("Too Much Data" by Acemoglu et al. 2022), the hypotheses are usually:
- H1: comparative_static — rho increases → equilibrium share_rate decreases
- H2: knowledge_dependent — more externality context in prompt → better mechanism understanding
- H3: dynamic_convergence — LLM converges to BNE through fictitious play belief updating

Each hypothesis must have:
- id, statement, nature (one of: comparative_static / knowledge_dependent / dynamic_convergence)
- preferred_test, parameters_to_vary, success_criterion

Before finalizing paradigm.yaml, you MUST:
1. Send a query to the Solver Builder asking if all hypotheses are numerically testable
2. Wait for confirmation or negotiate adjustments
3. Only write_paradigm once all hypotheses are confirmed testable

Use the available tools to query the Paper Agent for clarification and to write the final paradigm.
Your final action MUST be calling write_paradigm.
"""


class ScenarioExtractorAgent:
    def __init__(self, model: str | None = None):
        self.client = AgentClient(model=model)
        self._paper_parse: dict = {}

    # ------------------------------------------------------------------ #
    # Main entry                                                         #
    # ------------------------------------------------------------------ #

    def extract(self) -> dict:
        """
        Read paper_parse.json, extract paradigm, write paradigm.yaml.
        Returns the paradigm dict.
        """
        logger.info(f"[{AGENT_NAME}] Starting extraction")
        deadline = time.time() + 300
        while not fio.file_exists("paper_parse.json"):
            if time.time() > deadline:
                raise TimeoutError("paper_parse.json not available after 5 minutes")
            logger.info(f"[{AGENT_NAME}] Waiting for paper_parse.json...")
            time.sleep(2)
        self._paper_parse = fio.read_json("paper_parse.json")

        tools = self._build_tools()
        messages = [
            {
                "role": "user",
                "content": (
                    "Here is the structured paper parse. "
                    "Please extract the game structure and formalize the three hypotheses. "
                    "Query the Paper Agent if you need clarification on any section. "
                    "Then notify the Solver Builder to check testability. "
                    "Once you receive confirmation, write the final paradigm using write_paradigm.\n\n"
                    f"PAPER PARSE:\n{self._paper_parse}"
                ),
            }
        ]

        logger.info(f"[{AGENT_NAME}] LLM extraction attempt 1")
        self.client.run(
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=tools,
            tool_executor=self._execute_tool,
            temperature=0.2,
        )

        if not fio.file_exists("paradigm.yaml"):
            logger.warning(f"[{AGENT_NAME}] paradigm.yaml missing after attempt 1; running strict attempt 2")
            self.client.run(
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Strict mode: your response must end by calling write_paradigm exactly once. "
                            "If you need clarification, call query_paper_agent first; then call "
                            "notify_solver_builder and wait for confirmation; finally call write_paradigm."
                        ),
                    }
                ],
                tools=tools,
                tool_executor=self._execute_tool,
                temperature=0.1,
                max_iterations=20,
            )

        # Fallback: if Claude didn't write paradigm.yaml, write a default one
        if not fio.file_exists("paradigm.yaml"):
            logger.error(f"[{AGENT_NAME}] LLM did not call write_paradigm; using fallback paradigm")
            self._write_default_paradigm()

        result = fio.read_yaml("paradigm.yaml")
        if result.get("_warning"):
            logger.warning(f"[{AGENT_NAME}] paradigm.yaml written via fallback path")
        else:
            logger.info(f"[{AGENT_NAME}] paradigm.yaml written via LLM tool call")
        return result

    # ------------------------------------------------------------------ #
    # Handle messages from Solver Builder                               #
    # ------------------------------------------------------------------ #

    def handle_pending_messages(self) -> int:
        messages = fio.list_messages(to_agent=AGENT_NAME, status="pending")
        processed = 0
        for msg in messages:
            fio.mark_message(msg["id"], "processing")
            try:
                if msg["type"] in ("reply", "negotiate"):
                    # Solver Builder replied about testability
                    # Log it — coordinator will decide if we need another iteration
                    logger.info(f"[{AGENT_NAME}] Received from {msg['from']}: {msg['content'][:200]}")
                    fio.mark_message(msg["id"], "done")
                    processed += 1
                else:
                    fio.mark_message(msg["id"], "done")
            except Exception as e:
                logger.error(f"[{AGENT_NAME}] Error: {e}")
                fio.mark_message(msg["id"], "error")
        return processed

    def sign_off(self) -> None:
        """Mark this agent as ready for Gate 1."""
        fio.write_json("extractor_signoff.json", {
            "agent": AGENT_NAME,
            "signed": True,
            "timestamp": time.time(),
        })
        logger.info(f"[{AGENT_NAME}] Signed off for Gate 1")

    # ------------------------------------------------------------------ #
    # Tools                                                              #
    # ------------------------------------------------------------------ #

    def _build_tools(self) -> list[dict]:
        return [
            make_tool_spec(
                name="query_paper_agent",
                description="Ask the Paper Agent a question about the paper content.",
                parameters={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Specific question about the paper",
                        },
                        "sections": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Which sections to focus on, e.g. ['comparative_statics']",
                        },
                    },
                    "required": ["question"],
                },
            ),
            make_tool_spec(
                name="notify_solver_builder",
                description=(
                    "Send the hypothesis draft to the Solver Builder to check "
                    "whether each hypothesis is numerically testable."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "hypotheses_summary": {
                            "type": "string",
                            "description": "Brief description of H1/H2/H3 for the Solver Builder",
                        }
                    },
                    "required": ["hypotheses_summary"],
                },
            ),
            make_tool_spec(
                name="write_paradigm",
                description="Write the finalized paradigm to workspace/paradigm.yaml.",
                parameters={
                    "type": "object",
                    "properties": {
                        "scenario_id": {"type": "string"},
                        "paper": {"type": "string"},
                        "game_type": {"type": "string"},
                        "players": {
                            "type": "array",
                            "items": {"type": "object"},
                        },
                        "equilibrium_concept": {"type": "string"},
                        "key_parameters": {
                            "type": "array",
                            "items": {"type": "object"},
                        },
                        "hypotheses": {
                            "type": "array",
                            "items": {"type": "object"},
                        },
                    },
                    "required": [
                        "scenario_id", "paper", "game_type",
                        "players", "equilibrium_concept",
                        "key_parameters", "hypotheses",
                    ],
                },
            ),
        ]

    def _execute_tool(self, name: str, inputs: dict) -> str:
        if name == "query_paper_agent":
            msg_id = fio.send_message(
                from_agent=AGENT_NAME,
                to_agent="paper_agent",
                msg_type="query",
                content=inputs["question"],
            )
            # Wait for reply (synchronous in Plan A)
            return self._wait_for_reply(msg_id, timeout=120)

        if name == "notify_solver_builder":
            msg_id = fio.send_message(
                from_agent=AGENT_NAME,
                to_agent="solver_builder",
                msg_type="notify",
                content=inputs["hypotheses_summary"],
            )
            logger.info(f"[{AGENT_NAME}] tool notify_solver_builder called")
            return self._wait_for_reply(msg_id, timeout=180)

        if name == "write_paradigm":
            fio.write_yaml("paradigm.yaml", inputs)
            self.sign_off()
            logger.info(f"[{AGENT_NAME}] tool write_paradigm called and signoff written")
            return "paradigm.yaml written and Extractor signed off."

        return f"Unknown tool: {name}"

    def _wait_for_reply(self, original_msg_id: str, timeout: int = 120) -> str:
        """Poll for a reply to a specific message (Plan A synchronous wait).
        Checks by reply_to id regardless of status to avoid race with coordinator."""
        msg_dir = fio.WORKSPACE / "messages"
        deadline = time.time() + timeout
        while time.time() < deadline:
            for f in msg_dir.glob("*.json"):
                try:
                    import json
                    msg = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if msg.get("reply_to") == original_msg_id and msg.get("to") == AGENT_NAME:
                    return msg["content"]
            time.sleep(2)
        return f"[Timeout waiting for reply to {original_msg_id}]"

    # ------------------------------------------------------------------ #
    # Fallback paradigm (Scenario B hardcoded)                           #
    # ------------------------------------------------------------------ #

    def _write_default_paradigm(self) -> None:
        """Write the known-correct Scenario B paradigm if Claude didn't do it."""
        paradigm = {
            "scenario_id": "scenario_b",
            "paper": "Acemoglu et al. 2022 - Too Much Data",
            "game_type": "static_bayesian_game",
            "players": [
                {
                    "role": "user",
                    "count": "N",
                    "action_space": ["share", "not_share"],
                },
                {
                    "role": "platform",
                    "count": 1,
                    "action_space": "continuous_price_vector",
                },
            ],
            "equilibrium_concept": "Nash_BNE",
            "key_parameters": [
                {
                    "name": "rho",
                    "description": "inter-user type correlation",
                    "range": [0, 1],
                },
                {
                    "name": "v",
                    "description": "privacy preference value",
                    "range": "continuous",
                },
            ],
            "hypotheses": [
                {
                    "id": "H1",
                    "statement": "rho increases → equilibrium share_rate decreases",
                    "nature": "comparative_static",
                    "preferred_test": "SensitivityTest",
                    "parameters_to_vary": ["rho"],
                    "success_criterion": "EAS < 0 for all rho grid points",
                },
                {
                    "id": "H2",
                    "statement": "More externality context in prompt → better mechanism understanding",
                    "nature": "knowledge_dependent",
                    "preferred_test": "PromptLadderTest",
                    "success_criterion": "Jaccard monotone increasing v1→v6",
                },
                {
                    "id": "H3",
                    "statement": "LLM converges to BNE through fictitious play belief updating",
                    "nature": "dynamic_convergence",
                    "preferred_test": "FictitiousPlayTest",
                    "success_criterion": "strategy_delta < 0.01 within 50 rounds",
                },
            ],
            "_warning": "Fallback paradigm — Extractor Agent did not call write_paradigm",
        }
        fio.write_yaml("paradigm.yaml", paradigm)
        self.sign_off()
