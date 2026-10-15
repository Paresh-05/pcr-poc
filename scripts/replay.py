"""Run one diagnosis through the service with a fixed answer pattern and print the path. Makes real AI calls.

Usage: uv run --system-certs python -m scripts.replay P187973 MG_COMET_EV yes,no,yes,not_sure,yes,no
"""
import asyncio
import sys
import uuid

from app import diagnosis
from app.config import get_settings
from app.db import create_pool
from app.diagnosis_graph import build_graph
from app.llm import OpenAILLM
from app.usage import UsageRecorder


async def main(dtc: str, model: str, answers: list[str]) -> None:
    settings = get_settings()
    pool = create_pool(settings)
    await pool.open()
    try:
        llm = OpenAILLM(settings, UsageRecorder(pool))
        graph = build_graph(llm)
        d = await diagnosis.start_diagnosis(pool, llm, graph, dtc, model)
        did = uuid.UUID(d["diagnosis_id"])
        top = lambda d: ", ".join(f"{c['cause'][:38]} {c['likelihood']}%" for c in d["candidates"][:3])
        print(f"START  {top(d)}")
        for a in answers:
            if d["status"] != "asking":
                break
            print(f"\nQ: {d['question']}\nA: {a}")
            d = await diagnosis.answer_diagnosis(pool, llm, graph, did, a)
            print(f"   showed: {d['explanation']}\n   now: {top(d)}")
        k = d["conclusion"]
        print(f"\nSTATUS {d['status']} after {d['questions_asked']} answers")
        if k:
            print(f"PROBLEM: {k['problem']}\nCONFIDENCE: {k['confidence']} - {k['confidence_note']}")
            print("HOW:", *k["how_we_got_here"], sep="\n - ")
            print("RULED OUT:", *k["ruled_out"], sep="\n - ")
            print("UNCERTAIN:", *k["uncertain"], sep="\n - ")
            print("PARTS:", k["parts"])
        print("COST:", d["cost"], "| id", did)
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1].upper(), sys.argv[2].upper(), sys.argv[3].split(",")), loop_factory=asyncio.SelectorEventLoop)
