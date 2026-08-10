"""TimeOffBot — the app SHELL.

This is your starting point. The chat works: a message you type goes to an
LLM and comes back as plain conversation, and you can switch which employee
you're acting as.

The application now also has:

- RAG-based policy questions
- Country-aware policy retrieval
- Intent classification
- Balance lookup
"""

import json
from pathlib import Path

from fastapi import FastAPI, Header
from fastapi.responses import FileResponse
from openai import AzureOpenAI
from pydantic import BaseModel

from . import config
from .rag import search_policy
from .nodes.intent import classify_intent
from .tools.balance_tool import get_balance
from .tools.request_tool import get_requests
from .nodes.submit_leave_extractor import extract_leave_request
from .tools.submit_request_tool import submit_request


app = FastAPI(title="TimeOffBot")


_EMPLOYEES = {
    e["id"]: e
    for e in json.loads(
        (config.DATA_DIR / "employees.json").read_text()
    )
}


class ChatRequest(BaseModel):
    message: str


def _client() -> AzureOpenAI:
    """Create and return the Azure OpenAI client."""

    if not (config.AZURE_ENDPOINT and config.AZURE_API_KEY):
        raise RuntimeError(
            "Azure OpenAI credentials are missing. "
            "Copy .env.example to .env and fill in your values."
        )

    return AzureOpenAI(
        azure_endpoint=config.AZURE_ENDPOINT,
        api_key=config.AZURE_API_KEY,
        api_version=config.AZURE_API_VERSION,
    )


@app.get("/api/users")
def users():
    """The employees you can act as, shown in the UI's user switcher."""

    return list(_EMPLOYEES.values())


@app.post("/api/chat")
def chat(
    req: ChatRequest,
    x_user_id: str = Header(default="", alias="X-User-Id"),
):
    """Handle a chat message."""

    # ---------------------------------------------------------
    # 1. Identify the currently selected employee
    # ---------------------------------------------------------

    emp = _EMPLOYEES.get(x_user_id)

    who = (
        f"You are talking to {emp['full_name']}, "
        f"based in {emp['country']}."
        if emp
        else "You are talking to an Acme Corp employee."
    )

    # ---------------------------------------------------------
    # 2. Create the Azure OpenAI client
    # ---------------------------------------------------------

    client = _client()

    # ---------------------------------------------------------
    # 3. Determine what the user wants
    # ---------------------------------------------------------

    intent = classify_intent(client, req.message)

    # ---------------------------------------------------------
    # 4. Handle POLICY questions using RAG
    # ---------------------------------------------------------

    if intent == "POLICY":

        context = search_policy(
            req.message,
            country=emp["country"] if emp else None,
        )

        system = (
            "You are TimeOffBot, a helpful assistant for Acme Corp employees.\n\n"
            f"{who}\n\n"
            "Answer ONLY using the policy information provided below.\n"
            "If the answer is not present in the policy, say you don't know.\n\n"
            "POLICY INFORMATION:\n"
            f"{context}"
        )

    # ---------------------------------------------------------
    # 5. Handle BALANCE questions
    # ---------------------------------------------------------

    elif intent == "BALANCE":

        if not emp:

            system = (
                "I could not identify the currently selected employee. "
                "Please select an employee and try again."
            )

        else:

            balance_result = get_balance(emp["id"])

            if balance_result["success"]:

                balances = balance_result["balances"]

                balance_text = "\n".join(
                    f"- {balance['leave_type']}: "
                    f"{balance['remaining_days']} days remaining"
                    for balance in balances
                )

                system = (
                    "You are TimeOffBot, a helpful assistant for Acme Corp employees.\n\n"
                    f"{who}\n\n"
                    "The employee's actual leave balances are:\n"
                    f"{balance_text}\n\n"
                    "Answer the user's question using ONLY these balances. "
                    "Do not invent or modify any numbers."
                )

            else:

                system = (
                    "You are TimeOffBot. "
                    "No balance information was found for the currently "
                    "selected employee."
                )

   # ---------------------------------------------------------
# 6. Handle LIST_REQUESTS
# ---------------------------------------------------------

    elif intent == "LIST_REQUESTS":

        if not emp:

            system = (
                "I could not identify the currently selected employee. "
                "Please select an employee and try again."
            )


    elif intent == "SUBMIT_REQUEST":

        if not emp:

            system = (
                "I could not identify the currently selected employee. "
                "Please select an employee and try again."
            )

        else:

            extracted = extract_leave_request(
                client,
                config.AZURE_CHAT_DEPLOYMENT,
                req.message,
            )

            leave_type = extracted.get("leave_type")
            start_date = extracted.get("start_date")
            end_date = extracted.get("end_date")

            # -----------------------------------------------------
            # Check for missing information
            # -----------------------------------------------------

            missing = []

            if not leave_type:
                missing.append("leave type")

            if not start_date:
                missing.append("start date")

            if not end_date:
                missing.append("end date")

            if missing:

                system = (
                    "The user wants to submit a time-off request, "
                    "but some information is missing.\n\n"
                    f"Missing information: {', '.join(missing)}.\n\n"
                    "Ask the user to provide the missing information. "
                    "Do not submit a request yet."
                )

            else:

                # -------------------------------------------------
                # Submit the request using the actual tool
                # -------------------------------------------------

                result = submit_request(
                    employee_id=emp["id"],
                    leave_type=leave_type,
                    start_date=start_date,
                    end_date=end_date,
                )

                if result["success"]:

                    request = result["request"]

                    system = (
                        "The time-off request was successfully recorded.\n\n"
                        f"Request ID: {request['id']}\n"
                        f"Leave type: {request['leave_type']}\n"
                        f"Start date: {request['start_date']}\n"
                        f"End date: {request['end_date']}\n"
                        f"Status: {request['status']}\n\n"
                        "Tell the user that the request was submitted successfully. "
                        "Use ONLY the information above."
                    )

                else:

                    system = (
                        "The time-off request was NOT submitted.\n\n"
                        f"Reason: {result['message']}\n\n"
                        "Tell the user clearly that the request was not submitted. "
                        "Do not claim that it was successful."
                    )

    else:

        request_result = get_requests(emp["id"])

        if request_result["success"]:

            requests = request_result["requests"]

            if not requests:

                system = (
                    f"{who}\n\n"
                    "You currently have no time-off requests."
                )

            else:

                request_text = "\n".join(
                    f"- {request['id']}: "
                    f"{request['leave_type']} leave, "
                    f"{request['start_date']} to {request['end_date']}, "
                    f"status: {request['status']}"
                    for request in requests
                )

                system = (
                    "You are TimeOffBot, a helpful assistant for Acme Corp employees.\n\n"
                    f"{who}\n\n"
                    "The employee's actual time-off requests are:\n"
                    f"{request_text}\n\n"
                    "Answer using ONLY these requests. "
                    "Do not invent or modify any request information."
                )

        else:

            system = (
                "No time-off request information was found "
                "for the currently selected employee."
            )

    # ---------------------------------------------------------
    # 7. Ask Azure OpenAI to generate the final response
    # ---------------------------------------------------------

    resp = client.chat.completions.create(
        model=config.AZURE_CHAT_DEPLOYMENT,
        temperature=0.5,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": req.message},
        ],
    )

    return {
        "reply": resp.choices[0].message.content
    }


@app.get("/")
def index():
    return FileResponse(
        Path(__file__).parent / "index.html"
    )