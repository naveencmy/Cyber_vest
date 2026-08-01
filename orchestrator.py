import argparse
import asyncio
import os
import time
from uuid import uuid4
from urllib.parse import urlparse

from a2a.client.client_factory import create_client
from a2a.types.a2a_pb2 import (
    Message,
    Part,
    Role,
    SendMessageRequest,
)
from utils.logger_config import setup_logger

logger = setup_logger("Orchestrator")

_clients = {}


async def get_client_for_agent(base_url: str):
    if base_url in _clients:
        return _clients[base_url]
    logger.info(f"[Orchestrator] Discovering agent at {base_url}...")
    client = await create_client(base_url)
    _clients[base_url] = client
    logger.info(f"[Orchestrator] Found agent at {base_url}")
    return client


async def close_all_clients():
    """Clean up open client connections."""
    for base_url, client in _clients.items():
        if hasattr(client, "close"):
            if asyncio.iscoroutinefunction(client.close):
                await client.close()
            else:
                client.close()
    _clients.clear()


def _extract_text_from_event(event) -> str | None:
    """Extract text from a StreamResponse, handling message, task, and artifact formats."""
    if event.HasField("message"):
        parts = [p.text for p in event.message.parts if p.text]
        return "".join(parts) if parts else None

    if event.HasField("task"):
        task = event.task
        if task.HasField("status") and task.status.HasField("message"):
            parts = [p.text for p in task.status.message.parts if p.text]
            return "".join(parts) if parts else None

    if event.HasField("artifact_update"):
        artifact = event.artifact_update
        if artifact.HasField("artifact"):
            parts = [p.text for p in artifact.artifact.parts if p.text]
            return "".join(parts) if parts else None

    return None


async def _send_text_message(client, text: str) -> str:
    """Send a text message and accumulate all streamed text responses."""
    msg = Message()
    msg.message_id = str(uuid4())
    msg.role = Role.ROLE_USER
    part = msg.parts.add()
    part.text = text

    req = SendMessageRequest()
    req.message.CopyFrom(msg)

    for attempt in range(3):
        try:
            stream = client.send_message(req)
            received_chunks = []

            async for event in stream:
                resp_text = _extract_text_from_event(event)
                if resp_text:
                    received_chunks.append(resp_text)

            full_response = "".join(received_chunks)
            if full_response:
                return full_response

            raise RuntimeError("No text response received from agent")
        except Exception as e:
            logger.warning(f"[WARN] Retrying message send (attempt {attempt+1}/3): {e}")
            await asyncio.sleep(5 * (attempt + 1))

    raise RuntimeError("All retries failed")


async def run_scan(repo_url: str) -> str:
    try:
        scanner, analyzer, reporter = await asyncio.gather(
            get_client_for_agent(os.getenv("SCANNER_URL", "http://localhost:8001")),
            get_client_for_agent(os.getenv("ANALYZER_URL", "http://localhost:8002")),
            get_client_for_agent(os.getenv("REPORTER_URL", "http://localhost:8003")),
        )

        t0 = time.time()
        logger.info(f"\n[Orchestrator] Starting scan for {repo_url}")

        repo_digest = await _send_text_message(scanner, repo_url)
        logger.info(f"[1/3] Scanner complete ({len(repo_digest)} bytes) [{time.time()-t0:.1f}s]")

        vuln_json = await _send_text_message(analyzer, repo_digest)
        logger.info(f"[2/3] Analyzer complete ({len(vuln_json)} bytes) [{time.time()-t0:.1f}s]")

        report_md = await _send_text_message(reporter, vuln_json)
        logger.info(f"[3/3] Reporter complete [{time.time()-t0:.1f}s total]")
        return report_md
    finally:
        await close_all_clients()


def run_scan_sync(repo_url: str) -> str:
    return asyncio.run(run_scan(repo_url))


def _sanitize_filename(url: str) -> str:
    parsed = urlparse(url)
    clean_name = f"{parsed.netloc}{parsed.path}".strip("/").replace("/", "_").replace(":", "_")
    return clean_name or "report"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3-Agent Repo Security Scanner")
    parser.add_argument("--url", required=True, help="Git Repository URL to scan")
    args = parser.parse_args()

    report = run_scan_sync(args.url)

    logger.info("\n" + "="*60 + "\nFINAL SECURITY REPORT\n" + "="*60 + f"\n\n{report}")

    os.makedirs("reports", exist_ok=True)
    filename = f"reports/{_sanitize_filename(args.url)}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"[Orchestrator] Report saved to {filename}")