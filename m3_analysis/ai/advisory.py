import os
import json
from typing import Dict, Any, List, Optional
from groq import Groq
from utils.helpers import get_logger, timer_decorator

from config import GROQ_MODEL_NAME

logger = get_logger("AIDecisionSupport")

class AIAdvisoryGenerator:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes Groq API client if key is available.
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("GROQ_API_KEY not found in environment. Running in Mock Advisory mode.")

    @timer_decorator
    def generate_advisory(self, disaster_metrics: Dict[str, Any], route_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Queries Groq LLM (Llama 3) to generate disaster response recommendations.
        Gracefully falls back to mock reports if the API call fails or is unconfigured.
        """
        scenario = disaster_metrics.get("scenario_type", "Flood disaster")
        pre = disaster_metrics.get("metrics", {}).get("pre_disaster", {})
        post = disaster_metrics.get("metrics", {}).get("post_disaster", {})
        delta = disaster_metrics.get("metrics", {}).get("delta", {})

        route_summaries = []
        for r in route_analyses:
            route_summaries.append(
                f"- Route '{r['route_name']}': Status={r['status']}, Pre-distance={r.get('pre_disaster_distance', 0.0):.1f}m, "
                f"Post-distance={r.get('post_disaster_distance', 0.0):.1f}m, Detour={r['detour_ratio']:.2f}x"
            )
        routes_text = "\n".join(route_summaries)

        prompt = f"""
You are an Emergency Response Coordinator and Urban Planner.
Analyze the following satellite-derived road network simulation data and suggest mitigation advisories.

DISASTER PROFILE:
Scenario: {scenario}

NETWORK TOPOLOGY IMPACT:
- Global Network Efficiency Drop: {delta.get('efficiency_loss_percent', 0.0):.2f}% (Pre: {pre.get('efficiency', 0.0):.4f}, Post: {post.get('efficiency', 0.0):.4f})
- Newly Isolated Nodes: {delta.get('newly_isolated_nodes', 0)}
- Isolated Nodes Count: {post.get('isolated_nodes_count', 0)} (out of total network)

EMERGENCY CORRIDOR DISRUPTIONS:
{routes_text}

Provide your advisory strictly in JSON format matching the schema below:
{{
  "emergency_corridors_priority": [
    {{
      "corridor": "Name of the corridor",
      "status": "Short description of disruption level",
      "action": "Specific emergency traffic action required"
    }}
  ],
  "mitigation_strategies": [
    "High level strategy 1",
    "High level strategy 2"
  ]
}}
Do not write any introductory or concluding text, write only valid parseable JSON.
"""

        if not self.client:
            return self._get_mock_advisory(scenario, delta, route_analyses)

        try:
            logger.info("Querying Groq LLM for urban resilience advisory...")
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert AI urban resilience advisor. You always reply in raw, valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=GROQ_MODEL_NAME,
                response_format={"type": "json_object"},
                temperature=0.2
            )
            response_text = chat_completion.choices[0].message.content
            return json.loads(response_text)
        except Exception as e:
            logger.error(f"Failed to query Groq LLM: {e}. Falling back to mock advisory.")
            return self._get_mock_advisory(scenario, delta, route_analyses)

    def _get_mock_advisory(self, scenario: str, delta: Dict[str, Any], route_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Provides fallback mock advisory recommendations."""
        logger.info("Generating mock AI advisory report...")
        priority_corridors = []
        for r in route_analyses:
            if r['status'] in ["DISRUPTED_ISOLATED", "RE_ROUTED"]:
                priority_corridors.append({
                    "corridor": r['route_name'],
                    "status": f"Impacted: {r['status']} (detour ratio: {r['detour_ratio']:.2f}x)",
                    "action": "Reroute emergency response units using alternative local grid links. Deploy staging point at nearest intact node."
                })

        if not priority_corridors:
            priority_corridors.append({
                "corridor": "All Main Corridors",
                "status": "Stable/No immediate action required",
                "action": "Maintain standby status for emergency services."
            })

        loss = delta.get('efficiency_loss_percent', 0.0)
        mitigation = [
            f"Pre-position temporary physical barriers along flood-prone areas where nodes were severed.",
            f"Network efficiency degraded by {loss:.1f}%. Prioritize arterial road clearing to re-establish central backbone links.",
            "Deploy mobile cellular/radio links to coordinate staging hubs in newly isolated communities."
        ]

        return {
            "emergency_corridors_priority": priority_corridors,
            "mitigation_strategies": mitigation
        }
