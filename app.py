"""Streamlit app on one page: KPIs, carrier quote, LLM advisor, carrier charts. Start the API first.
Run:  API_URL=http://localhost:8010 streamlit run app.py --server.port 8510
"""
import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from data import iqr_upper_fence, run_query

API_URL = os.getenv("API_URL", "http://localhost:8010")
BLUE = "#2a78d6"
st.set_page_config(page_title="Carrier Advisor", layout="wide")


@st.cache_data
def load_data():
    """Shipments with the 4 data-entry costs removed (same 3 x IQR rule as the models)."""
    df = run_query("SELECT * FROM shipments_clean")
    df.loc[df["Cost"] > iqr_upper_fence(df["Cost"].dropna()), "Cost"] = None
    return df


def carrier_scorecard(df):
    done = df[df["Status"] != "In Transit"].copy()
    done["miles_with_cost"] = done["Distance_miles"].where(done["Cost"].notna())
    table = done.groupby("Carrier").agg(shipments=("Shipment_ID", "count"), spend_usd=("Cost", "sum"),
                                        miles=("miles_with_cost", "sum"), exception_pct=("is_exception", "mean"))
    table["cost_per_mile_usd"] = (table["spend_usd"] / table["miles"]).round(3)
    table["exception_pct"] = (100 * table["exception_pct"]).round(1)
    return table.drop(columns="miles").round(2).reset_index()


def bar(df, x, title, label):
    fig = px.bar(df.sort_values(x), x=x, y="Carrier", orientation="h", title=title,
                 color_discrete_sequence=[BLUE], labels={x: label, "Carrier": ""})
    return fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))


def kpis(df, table):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Shipments", f"{len(df):,}")
    c2.metric("Spend (USD)", f"${df['Cost'].sum():,.0f}")
    c3.metric("Cheapest per mile", table.sort_values("cost_per_mile_usd")["Carrier"].iloc[0])
    c4.metric("Exception rate", f"{100 * df.loc[df['Status'] != 'In Transit', 'is_exception'].mean():.1f}%")


def quote_panel():
    st.subheader("Carrier quote")
    lists = run_query("SELECT DISTINCT Origin_Warehouse, Destination FROM shipments_clean")
    origins, cities = sorted(lists["Origin_Warehouse"].unique()), sorted(lists["Destination"].unique())
    with st.form("quote"):
        a, b = st.columns(2)
        origin = a.selectbox("Origin warehouse", origins, index=origins.index("Warehouse_LA"))
        city = b.selectbox("Destination", cities, index=cities.index("Boston"))
        c, d = st.columns(2)
        weight = c.number_input("Weight (kg)", min_value=0.1, value=20.0)
        model = d.radio("Model", ["ensemble", "xgboost", "keras"], horizontal=True,
                        help="ensemble = average of XGBoost and the Keras neural network (most accurate)")
        if st.form_submit_button("Compare carriers"):
            payload = {"origin_warehouse": origin, "destination": city, "weight_kg": weight, "model": model}
            st.session_state.quote_result = requests.post(f"{API_URL}/quote", json=payload, timeout=30).json()
    result = st.session_state.get("quote_result")
    if result and "quotes" in result:
        quotes = pd.DataFrame(result["quotes"])
        st.success(f"Cheapest: **{result['cheapest']}** · USD {quotes['predicted_cost_usd'].iloc[0]:.2f} "
                   f"({quotes['distance_miles'].iloc[0]:,.0f} miles, model estimate)")
        st.dataframe(quotes.drop(columns="distance_miles"), hide_index=True, width="stretch")
    elif result:
        st.error(result.get("detail"))


def advisor_panel():
    st.subheader("Ask the advisor")
    with st.form("ask"):
        question = st.text_input("Question", placeholder="Ship 25 kg from LA to Boston, cheapest option")
        st.caption("Examples: a booking request, or 'Which carrier had the most delays in December?'")
        if st.form_submit_button("Ask") and question:
            resp = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=120)
            st.session_state.answer_result = resp.json() if resp.status_code == 200 else {"error": resp.json().get("detail")}
    answer = st.session_state.get("answer_result")
    if answer and "answer" in answer:
        with st.container(border=True):
            st.markdown(answer["answer"])
            st.caption(f"Tools used: {', '.join(answer['tools_used']) or 'none'}")
    elif answer:
        st.error(answer["error"])


df = load_data()
table = carrier_scorecard(df)
st.title("🚚 Carrier Advisor")
st.caption("Predicted freight cost for every carrier before booking · costs in USD (model estimates)")
kpis(df, table)
left, right = st.columns(2)
with left:
    quote_panel()
with right:
    advisor_panel()
st.subheader("Carrier performance (2023)")
c1, c2 = st.columns(2)
c1.plotly_chart(bar(table, "cost_per_mile_usd", "Cost per mile", "USD per mile"), width="stretch")
c2.plotly_chart(bar(table, "exception_pct", "Exception rate (not significantly different)", "%"), width="stretch")
st.dataframe(table, hide_index=True, width="stretch")
