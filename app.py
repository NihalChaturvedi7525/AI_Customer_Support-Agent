
import os
from getpass import getpass
from typing import TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

# Groq API Key
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass("Enter your Groq API Key: ")

# LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# State
class CustomerState(TypedDict):
    user_input: str
    category: str
    response: str

# Query Classification
def classify_query(state):
    message = state["user_input"].lower()

    if "order" in message or "delivery" in message:
        category = "Order"

    elif "refund" in message or "money back" in message:
        category = "Refund"

    elif "complaint" in message or "angry" in message or "damaged" in message:
        category = "Complaint"

    else:
        category = "General"

    return {"category": category}

# Order Support
def order_support(state):
    prompt = f"""
You are a helpful customer support agent.

Customer message:
{state["user_input"]}

Order status:
Your order is currently out for delivery.

Give a short and polite response.
"""

    result = llm.invoke(prompt)

    return {"response": result.content}

# Refund Support
def refund_support(state):
    prompt = f"""
You are a customer support agent.

Customer message:
{state["user_input"]}

Explain the refund process simply and politely.
"""

    result = llm.invoke(prompt)

    return {"response": result.content}

# Complaint Support
def complaint_support(state):
    prompt = f"""
You are a professional customer support agent.

Customer complaint:
{state["user_input"]}

Apologize politely and provide a helpful response.
"""

    result = llm.invoke(prompt)

    return {
        "response": result.content +
        "\n\nYour issue has been escalated to a human support agent."
    }

# General Support
def general_support(state):
    prompt = f"""
You are a helpful customer support agent.

Customer message:
{state["user_input"]}

Answer politely in simple language.
"""

    result = llm.invoke(prompt)

    return {"response": result.content}

# Routing
def route_query(state):

    if state["category"] == "Order":
        return "order"

    elif state["category"] == "Refund":
        return "refund"

    elif state["category"] == "Complaint":
        return "complaint"

    else:
        return "general"

# LangGraph
graph = StateGraph(CustomerState)

graph.add_node("classify", classify_query)
graph.add_node("order", order_support)
graph.add_node("refund", refund_support)
graph.add_node("complaint", complaint_support)
graph.add_node("general", general_support)

graph.add_edge(START, "classify")

graph.add_conditional_edges(
    "classify",
    route_query,
    {
        "order": "order",
        "refund": "refund",
        "complaint": "complaint",
        "general": "general"
    }
)

graph.add_edge("order", END)
graph.add_edge("refund", END)
graph.add_edge("complaint", END)
graph.add_edge("general", END)

app = graph.compile()

# Chatbot
print("AI Customer Support Agent Ready!")
print("Type 'exit' to stop.")

while True:

    user_message = input("You: ")

    if user_message.lower() == "exit":
        print("Chat ended.")
        break

    result = app.invoke({
        "user_input": user_message,
        "category": "",
        "response": ""
    })

    print("Category:", result["category"])
    print("AI:", result["response"])
