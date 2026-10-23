"""All project checks in one file. Run from the project root:  python -m testing.test_project
(after python data.py and python train.py). The live Groq check runs only when GROQ_API_KEY is set.
"""
import os

import pandas as pd
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from agent import ask, build_agent, run_sql_tool
from api import app, llm_ready
from data import clean_features, run_query, run_safe_select, training_rows
from predict import quote

client = TestClient(app)


def test_database():
    assert run_query("SELECT COUNT(*) AS n FROM shipments_clean")["n"].iloc[0] == 2000


def test_sql_guard_blocks_writes():
    for sql in ["DELETE FROM shipments", "SELECT 1; DROP TABLE shipments"]:
        try:
            run_safe_select(sql)
            raise AssertionError(f"not blocked: {sql}")
        except ValueError:
            pass


def test_cleaning_rules():
    rules = {"weight_cap_kg": 100.0, "cost_cap_usd": 700.0}
    df = pd.DataFrame({"Weight_kg": [5404.0, 20.0, 10.0], "Cost": [100.0, None, 6500.0]})
    assert clean_features(df, rules)["Weight_kg"].iloc[0] == 100.0
    assert len(training_rows(df, rules)) == 1


def test_quote_all_models():
    for model in ["ensemble", "xgboost", "keras"]:
        q = quote("Warehouse_LA", "Boston", 25, model=model)
        assert len(q) == 7 and q["predicted_cost_usd"].is_monotonic_increasing


def test_api_quote_and_validation():
    good = {"origin_warehouse": "Warehouse_LA", "destination": "Boston", "weight_kg": 25}
    assert client.post("/quote", json=good).status_code == 200
    assert client.post("/quote", json={**good, "weight_kg": -1}).status_code == 422
    assert client.post("/quote", json={**good, "origin_warehouse": "Mars"}).status_code == 422


def test_agent_uses_tool_with_fake_llm():
    call = {"name": "quote_carriers_tool", "id": "1",
            "args": {"origin_warehouse": "Warehouse_LA", "destination": "Boston", "weight_kg": 25}}
    fake = GenericFakeChatModel(messages=iter([AIMessage(content="", tool_calls=[call]),
                                               AIMessage(content="Book USPS.")]))
    object.__setattr__(fake, "bind_tools", lambda *args, **kwargs: fake)
    result = ask("Ship 25 kg from LA to Boston", agent=build_agent(fake))
    assert result["tools_used"] == ["quote_carriers_tool"]
    assert run_sql_tool.invoke({"query": "DROP TABLE shipments"}).startswith("ERROR")


def test_live_groq():
    if not llm_ready():
        print("SKIP  live Groq check (no GROQ_API_KEY)")
        return
    result = ask("Ship 25 kg from LA to Boston, cheapest option")
    print(result["answer"][:300])
    assert "quote_carriers_tool" in result["tools_used"]


if __name__ == "__main__":
    tests = [(n, f) for n, f in dict(globals()).items() if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\n{len(tests)} checks passed")
