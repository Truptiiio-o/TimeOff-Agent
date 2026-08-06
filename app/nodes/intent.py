from openai import AzureOpenAI

from .. import config


INTENTS = [
    "POLICY",
    "BALANCE",
    "SUBMIT_REQUEST",
    "LIST_REQUESTS",
    "UNSUPPORTED",
]


def classify_intent(client: AzureOpenAI, message: str) -> str:
    """
    Classify an employee's message into one of the supported intents.
    """

    system_prompt = """
You are an intent classifier for Acme Corp's Time-Off Management Agent.

Classify the user's message into exactly ONE of these intents:

POLICY
- Questions about leave policies, rules, eligibility, allowances,
  carryover, notice periods, sick leave, parental leave, etc.

BALANCE
- Questions asking how much leave the employee currently has remaining.

SUBMIT_REQUEST
- The employee wants to create or submit a new time-off request.

LIST_REQUESTS
- The employee wants to see their existing or previous time-off requests.

UNSUPPORTED
- Anything unrelated to time-off management.

Return ONLY the intent name.
Do not explain your answer.
"""

    response = client.chat.completions.create(
        model=config.AZURE_CHAT_DEPLOYMENT,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
    )

    intent = response.choices[0].message.content.strip().upper()

    if intent not in INTENTS:
        return "UNSUPPORTED"

    return intent


if __name__ == "__main__":

    client = AzureOpenAI(
        azure_endpoint=config.AZURE_ENDPOINT,
        api_key=config.AZURE_API_KEY,
        api_version=config.AZURE_API_VERSION,
    )

    test_messages = [
        "How many annual leave days do I get?",
        "How much leave do I have left?",
        "I want to take leave from August 20 to August 22.",
        "Show me my previous time-off requests.",
        "What is the weather today?",
    ]

    for message in test_messages:
        intent = classify_intent(client, message)

        print(f"\nMessage: {message}")
        print(f"Intent:  {intent}")