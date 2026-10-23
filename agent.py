"""LangChain agent with two tools: price a shipment (model) and look up history (SQL).

The LLM reads a plain-English request, picks a tool, and answers only with numbers
the tools returned.
"""
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from data import run_safe_select
from predict import options, quote

load_dotenv()


@tool
def quote_carriers_tool(origin_warehouse: str, destination: str, weight_kg: float) -> str:
    """Predicted cost in USD for ALL carriers on one shipment, cheapest first.
    origin_warehouse must be a code like Warehouse_LA; destination a city like Boston."""
    try:
        return quote(origin_warehouse, destination, weight_kg).to_csv(index=False)
    except ValueError as err:
        return f"ERROR: {err}"


@tool
def run_sql_tool(query: str) -> str:
    """One read-only SQLite SELECT on table shipments_clean. Returns up to 20 rows as CSV."""
    try:
        return run_safe_select(query).to_csv(index=False)
    except Exception as err:  # bad SQL goes back to the LLM so it can fix it
        return f"ERROR: {err}"


def system_prompt():
    o = options()
    return f"""You are Carrier Advisor, an assistant for a US shipper's logistics team.
Tools: quote_carriers_tool prices a new shipment with every carrier (machine learning model).
run_sql_tool runs one SELECT on shipments_clean(Shipment_ID, Origin_Warehouse, Destination,
Carrier, Shipment_Date 'YYYY-MM-DD', Weight_kg, Cost, Status, Distance_miles, Transit_Days,
is_delayed 0/1, is_exception 0/1).
Warehouses: {o['Origin_Warehouse']}. Cities: {o['Destination']}. Carriers: {o['Carrier']}.
Rules: every number must come from a tool. Map a city to its warehouse code (LA -> Warehouse_LA).
Ask for the weight if it is missing. Costs are USD model estimates. Past exception rates do not
differ significantly between carriers, so recommend on price. Decline questions not about shipping.
Answer briefly: the recommendation first, then a small markdown table."""


def build_agent(model=None):
    """The app uses Groq; tests pass a fake model."""
    if model is None:
        from langchain_groq import ChatGroq
        model = ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"), temperature=0, max_tokens=900)
    return create_agent(model, tools=[quote_carriers_tool, run_sql_tool], system_prompt=system_prompt())


def ask(question, agent=None):
    """Answer one question. Returns the answer and which tools were used."""
    result = (agent or build_agent()).invoke({"messages": [{"role": "user", "content": question}]})
    answer = [m for m in result["messages"] if isinstance(m, AIMessage)][-1].content
    tools = [m.name for m in result["messages"] if isinstance(m, ToolMessage)]
    return {"answer": answer, "tools_used": tools}
