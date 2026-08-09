from typing import Annotated, TypedDict

from ..nodes.intent import classify_intent
from ..tools.balance_tool import get_balance
from ..tools.request_tool import get_requests
from ..nodes.submit_leave_extractor import extract_leave_request
from ..tools.request_tool import submit_request


from datetime import date
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command

from .. import config
from ..rag import search_policy

from openai import AzureOpenAI



class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

    employee_id: str
    country: str
    intent: str

    leave_type: str
    start_date: str
    end_date: str

    needs_clarification: bool
    confirmed: bool
    validation_passed: bool
    requested_days: int

    iteration: int
    response: str

def get_client():
    return AzureOpenAI(
        azure_endpoint=config.AZURE_ENDPOINT,
        api_key=config.AZURE_API_KEY,
        api_version=config.AZURE_API_VERSION,
    )

def intent_node(state: AgentState):

    print("🧠 Classifying intent...")

    client = get_client()

    message = state["messages"][-1].content

    intent = classify_intent(
        client,
        message,
    )

    print(f"🎯 Intent: {intent}")

    return {
        "intent": intent,
        "iteration": state.get("iteration", 0) + 1,
    }


def route_by_intent(state: AgentState):
    """
    Decide which node should handle the user's request.
    """

    intent = state["intent"]

    print(f"🔀 Routing based on intent: {intent}")

    if intent == "POLICY":
        return "policy"

    elif intent == "BALANCE":
        return "balance"

    elif intent == "LIST_REQUESTS":
        return "list_requests"

    elif intent == "SUBMIT_REQUEST":
        return "submit_request"

    else:
        return "unsupported"



def receive_request(state: AgentState):
    """
    Initial node that receives the user's request.
    """

    print("📥 Received request")

    return {
        "iteration": state.get("iteration", 0) + 1
    }


#####################################################

def policy_node(state: AgentState):

    print("📚 → POLICY node")

    client = get_client()

    message = state["messages"][-1].content

    context = search_policy(
        message,
        country=state["country"],
    )

    system = (
        "You are TimeOffBot, a helpful assistant for Acme Corp employees.\n\n"
        f"The employee is based in {state['country']}.\n\n"
        "Answer ONLY using the policy information provided below.\n"
        "If the answer is not present in the policy, say you don't know.\n\n"
        "POLICY INFORMATION:\n"
        f"{context}"
    )

    response = client.chat.completions.create(
        model=config.AZURE_CHAT_DEPLOYMENT,
        temperature=0.5,
        messages=[
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": message,
            },
        ],
    )

    answer = response.choices[0].message.content

    return {
        "response": answer,
        "iteration": state.get("iteration", 0) + 1,
    }


#####################################################

def balance_node(state: AgentState):

    print("💰 → BALANCE node")

    client = get_client()

    employee_id = state["employee_id"]

    balance_result = get_balance(employee_id)

    if not balance_result["success"]:

        return {
            "response": (
                "I could not find leave balance information "
                "for the selected employee."
            ),
            "iteration": state.get("iteration", 0) + 1,
        }

    balances = balance_result["balances"]

    balance_text = "\n".join(
        f"- {balance['leave_type']}: "
        f"{balance['remaining_days']} days remaining"
        for balance in balances
    )

    message = state["messages"][-1].content

    system = (
        "You are TimeOffBot, a helpful assistant for Acme Corp employees.\n\n"
        f"The employee ID is {employee_id}.\n\n"
        "The employee's actual leave balances are:\n"
        f"{balance_text}\n\n"
        "Answer the user's question using ONLY these balances.\n"
        "Do not invent, change, or calculate any different numbers."
    )

    response = client.chat.completions.create(
        model=config.AZURE_CHAT_DEPLOYMENT,
        temperature=0.5,
        messages=[
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": message,
            },
        ],
    )

    answer = response.choices[0].message.content

    return {
        "response": answer,
        "iteration": state.get("iteration", 0) + 1,
    }


####################################################

def list_requests_node(state: AgentState):

    print("📋 → LIST_REQUESTS node")

    client = get_client()

    employee_id = state["employee_id"]

    result = get_requests(employee_id)

    if not result["success"]:

        return {
            "response": (
                "I could not find any time-off request information "
                "for the selected employee."
            ),
            "iteration": state.get("iteration", 0) + 1,
        }

    requests = result["requests"]

    if not requests:

        return {
            "response": "You currently have no time-off requests.",
            "iteration": state.get("iteration", 0) + 1,
        }

    request_text = "\n".join(
        f"- {request['id']}: "
        f"{request['leave_type']} leave, "
        f"{request['start_date']} to {request['end_date']}, "
        f"status: {request['status']}"
        for request in requests
    )

    message = state["messages"][-1].content

    system = (
        "You are TimeOffBot, a helpful assistant for Acme Corp employees.\n\n"
        f"The employee ID is {employee_id}.\n\n"
        "The employee's actual time-off requests are:\n"
        f"{request_text}\n\n"
        "Answer the user's question using ONLY these requests. "
        "Do not invent or modify request information."
    )

    response = client.chat.completions.create(
        model=config.AZURE_CHAT_DEPLOYMENT,
        temperature=0.5,
        messages=[
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": message,
            },
        ],
    )

    answer = response.choices[0].message.content

    return {
        "response": answer,
        "iteration": state.get("iteration", 0) + 1,
    }

###################################################

def extract_request_node(state: AgentState):

    print("📝 → EXTRACT_REQUEST node")

    client = get_client()

    message = state["messages"][-1].content

    extracted = extract_leave_request(
        client,
        config.AZURE_CHAT_DEPLOYMENT,
        message,
    )

    leave_type = extracted.get("leave_type") or ""
    start_date = extracted.get("start_date") or ""
    end_date = extracted.get("end_date") or ""

    print(f"Leave type: {leave_type}")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")

    missing = []

    if not leave_type:
        missing.append("leave type")

    if not start_date:
        missing.append("start date")

    if not end_date:
        missing.append("end date")

    return {
        "leave_type": leave_type,
        "start_date": start_date,
        "end_date": end_date,
        "needs_clarification": len(missing) > 0,
        "iteration": state.get("iteration", 0) + 1,
    }


#####################################################

def route_after_extraction(state: AgentState):

    if state["needs_clarification"]:
        print("❓ Missing information → clarification")
        return "clarify"

    print("✅ All request details present → validation")
    return "validate"


######################################################


def clarification_node(state: AgentState):

    print("❓ → CLARIFICATION node")

    missing = []

    if not state["leave_type"]:
        missing.append("leave type")

    if not state["start_date"]:
        missing.append("start date")

    if not state["end_date"]:
        missing.append("end date")

    response = (
        "I need a little more information before I can submit "
        "your leave request. Please provide: "
        + ", ".join(missing)
        + "."
    )

    return {
        "response": response,
        "iteration": state.get("iteration", 0) + 1,
    }




def validation_node(state: AgentState):

    print("✅ → VALIDATION node")

    leave_type = state["leave_type"]
    start_date = state["start_date"]
    end_date = state["end_date"]
    employee_id = state["employee_id"]

    # ---------------------------------------------------------
    # 1. Validate date format
    # ---------------------------------------------------------

    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)

    except ValueError:

        return {
            "response": (
                "The dates provided are not valid. "
                "Please use the YYYY-MM-DD format."
            ),
        "validation_passed": False,
        "iteration": state.get("iteration", 0) + 1,
        }

    # ---------------------------------------------------------
    # 2. Check date order
    # ---------------------------------------------------------

    if end < start:

        return {
            "response": (
                "The end date cannot be before the start date."
            ),
            "validation_passed": False,
            "iteration": state.get("iteration", 0) + 1,
        }

    # ---------------------------------------------------------
    # 3. Calculate requested days
    # ---------------------------------------------------------

    requested_days = (end - start).days + 1

    print(f"📅 Requested days: {requested_days}")

    # ---------------------------------------------------------
    # 4. Check actual employee balance
    # ---------------------------------------------------------

    balance_result = get_balance(
        employee_id,
        leave_type,
    )

    if not balance_result["success"]:

        return {
            "response": (
                f"No balance information was found for "
                f"{leave_type} leave."
            ),
            "validation_passed": False,
            "iteration": state.get("iteration", 0) + 1,
        }

    remaining_days = balance_result["remaining_days"]

    print(f"💰 Remaining balance: {remaining_days}")

    # ---------------------------------------------------------
    # 5. Check sufficient balance
    # ---------------------------------------------------------

    if requested_days > remaining_days:

        return {
            "response": (
                f"You requested {requested_days} days of "
                f"{leave_type} leave, but only "
                f"{remaining_days} days are available."
            ),
            "validation_passed": False,
            "iteration": state.get("iteration", 0) + 1,
        }

    # ---------------------------------------------------------
    # 6. Validation successful
    # ---------------------------------------------------------

    print("✅ Validation successful")

    return {
    "response": (
        f"Your request for {requested_days} days of "
        f"{leave_type} leave is valid. "
        f"You have {remaining_days} days available."
    ),
    "validation_passed": True,
    "requested_days": requested_days,
    "iteration": state.get("iteration", 0) + 1,
    }



#########################################################


def route_after_validation(state: AgentState):

    if state["validation_passed"]:
        print("✅ Validation passed → confirmation")
        return "confirmation"

    print("❌ Validation failed → end")
    return "end"


########################################################


def confirmation_node(state: AgentState):

    print("🔔 → CONFIRMATION node")

    confirmation_message = (
        f"Your leave request is ready:\n\n"
        f"Leave type: {state['leave_type']}\n"
        f"Start date: {state['start_date']}\n"
        f"End date: {state['end_date']}\n"
        f"Number of days: {state['requested_days']}\n\n"
        "Would you like me to submit this request?"
    )

    # Pause the graph and wait for the user's answer
    answer = interrupt(confirmation_message)

    print(f"👤 Confirmation response: {answer}")

    normalized_answer = str(answer).strip().lower()

    confirmed = normalized_answer in [
        "yes",
        "y",
        "yeah",
        "yep",
        "confirm",
        "confirmed",
        "submit",
    ]

    if confirmed:

        return {
            "confirmed": True,
            "response": "Confirmation received. Submitting your request...",
            "iteration": state.get("iteration", 0) + 1,
        }

    else:

        return {
            "confirmed": False,
            "response": "Okay, I will not submit the leave request.",
            "iteration": state.get("iteration", 0) + 1,
        }


#########################################################


def route_after_confirmation(state: AgentState):

    if state["confirmed"]:
        print("✅ User confirmed → submitting")
        return "submit"

    print("❌ User declined → ending")
    return "end"


#######################################################

def submit_node(state: AgentState):

    print("🚀 → SUBMIT node")

    result = submit_request(
        employee_id=state["employee_id"],
        leave_type=state["leave_type"],
        start_date=state["start_date"],
        end_date=state["end_date"],
    )

    if not result["success"]:

        return {
            "response": (
                "I could not submit your leave request. "
                "Please try again."
            ),
            "iteration": state.get("iteration", 0) + 1,
        }

    return {
        "response": (
            f"Your {state['leave_type']} leave request has been "
            f"submitted successfully."
        ),
        "iteration": state.get("iteration", 0) + 1,
    }


#######################################################




def unsupported_node(state: AgentState):
    print("❓ → UNSUPPORTED node")
    return {}



builder = StateGraph(AgentState)

# -----------------------------
# Register nodes
# -----------------------------

builder.add_node("receive_request", receive_request)
builder.add_node("intent", intent_node)

builder.add_node("policy", policy_node)
builder.add_node("balance", balance_node)
builder.add_node("list_requests", list_requests_node)

builder.add_node("extract_request", extract_request_node)
builder.add_node("clarify", clarification_node)
builder.add_node("validate", validation_node)

builder.add_node("confirmation", confirmation_node)
builder.add_node("submit", submit_node)

builder.add_node("unsupported", unsupported_node)


# -----------------------------
# Initial flow
# -----------------------------

builder.add_edge(START, "receive_request")
builder.add_edge("receive_request", "intent")


# -----------------------------
# Intent routing
# -----------------------------

builder.add_conditional_edges(
    "intent",
    route_by_intent,
    {
        "policy": "policy",
        "balance": "balance",
        "list_requests": "list_requests",
        "submit_request": "extract_request",
        "unsupported": "unsupported",
    },
)


# -----------------------------
# Extraction routing
# -----------------------------

builder.add_conditional_edges(
    "extract_request",
    route_after_extraction,
    {
        "clarify": "clarify",
        "validate": "validate",
    },
)


# -----------------------------
# Validation routing
# -----------------------------

builder.add_conditional_edges(
    "validate",
    route_after_validation,
    {
        "confirmation": "confirmation",
        "end": END,
    },
)


builder.add_conditional_edges(
    "confirmation",
    route_after_confirmation,
    {
        "submit": "submit",
        "end": END,
    },
)


# -----------------------------
# End points
# -----------------------------

builder.add_edge("policy", END)
builder.add_edge("balance", END)
builder.add_edge("list_requests", END)
builder.add_edge("clarify", END)
builder.add_edge("submit", END)
builder.add_edge("unsupported", END)


memory = MemorySaver()

graph = builder.compile(
    checkpointer=memory
)


########################################################


def run_agent(
    message: str,
    employee_id: str,
    country: str,
    thread_id: str,
):
    """
    Run the TimeOffBot LangGraph for one user message.

    If the conversation is waiting for confirmation,
    resume the existing graph state.
    Otherwise, start a new graph execution.
    """

    run_config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # ---------------------------------------------------------
    # Check whether this conversation is currently paused
    # ---------------------------------------------------------

    snapshot = graph.get_state(run_config)

    is_interrupted = any(
        task.interrupts
        for task in snapshot.tasks
    )

    # ---------------------------------------------------------
    # Resume an interrupted conversation
    # ---------------------------------------------------------

    if is_interrupted:

        print("▶️ Resuming existing conversation...")

        result = graph.invoke(
            Command(resume=message),
            config=run_config,
        )

    # ---------------------------------------------------------
    # Start a new conversation
    # ---------------------------------------------------------

    else:

        print("🆕 Starting new conversation...")

        result = graph.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": message,
                    }
                ],
                "employee_id": employee_id,
                "country": country,
                "intent": "",

                "leave_type": "",
                "start_date": "",
                "end_date": "",

                "needs_clarification": False,
                "confirmed": False,
                "validation_passed": False,
                "requested_days": 0,

                "iteration": 0,
                "response": "",
            },
            config=run_config,
        )

    # ---------------------------------------------------------
    # Handle an interrupt / confirmation request
    # ---------------------------------------------------------

    interrupts = result.get("__interrupt__")

    if interrupts:

        return {
            "reply": interrupts[0].value,
            "interrupted": True,
        }

    # ---------------------------------------------------------
    # Normal response
    # ---------------------------------------------------------

    return {
        "reply": result.get(
            "response",
            "Sorry, I could not generate a response.",
        ),
        "interrupted": False,
    }



if __name__ == "__main__":

    thread_id = "E001-test-multiturn"

    print("\n--- FIRST MESSAGE ---")

    result = run_agent(
        message="I want annual leave from August 25 to August 27.",
        employee_id="E001",
        country="US",
        thread_id=thread_id,
    )

    print("\nBot:")
    print(result)


    print("\n--- SECOND MESSAGE ---")

    result = run_agent(
        message="Yes",
        employee_id="E001",
        country="US",
        thread_id=thread_id,
    )

    print("\nBot:")
    print(result)