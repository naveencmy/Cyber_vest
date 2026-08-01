import os
import sys
import pydantic
import uvicorn
from dotenv import load_dotenv

# Allow arbitrary types for Pydantic
pydantic.ConfigDict.arbitrary_types_allowed = True
pydantic.BaseModel.model_config = {"arbitrary_types_allowed": True}

sys.path.append("..")

from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from utils.logger_config import setup_logger

load_dotenv()
logger = setup_logger("ReporterAgent")

# Verify API Key availability
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "No Gemini API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY in your environment or .env file."
    )


REPORT_FORMATTER_INSTRUCTION = """
You are a technical security report writer.

Input:
- A JSON object (or JSON string) produced by a vulnerability analysis agent.
- It contains a `repo_summary` and a list of `findings` (id, title, severity, file, line_hint, description, recommendation).

Task:
- Convert this JSON into a clear, developer-friendly Markdown report.

Formatting rules:
1. Start with a top-level title: "# Security Report"
2. Create an "Overview" section:
   - Overall risk level with an emoji:
     - CRITICAL / HIGH -> 🚨
     - MEDIUM -> 🟡
     - LOW -> ✅
   - The `short_overview` text.
3. Create a "Findings" section:
   - If `findings` is empty, write: "No issues detected."
   - Otherwise, for each finding:
     - Heading: `## [<id>] <title>`
     - Severity emoji + Severity Level (e.g., 🚨 **HIGH**)
     - File location: `File:` `<file>` (`<line_hint>`)
     - Bullet points:
       - **Description:** <description>
       - **Recommendation:** <recommendation>

CRITICAL Output Rules:
- Output ONLY pure Markdown text.
- Do NOT wrap the Markdown in extra triple backticks (e.g., do NOT use ```markdown ... ```).
- Do NOT add conversational preambles or explanations outside the report.
"""


root_agent = Agent(
    name="reporter_agent",
    model="gemini-3.5-flash",
    description="Formats a repo vulnerability JSON into a developer-friendly Markdown report.",
    instruction=REPORT_FORMATTER_INSTRUCTION,
    tools=[],
)

# Expose as A2A server
a2a_app = to_a2a(root_agent, port=8003)


if __name__ == "__main__":
    logger.info("Starting ReporterAgent on port 8003...")
    uvicorn.run(a2a_app, host="0.0.0.0", port=8003)