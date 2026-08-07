import json

from openai import AzureOpenAI


def extract_leave_request(
    client: AzureOpenAI,
    deployment: str,
    message: str,
):
    """
    Extract leave type and dates from a user's natural-language request.
    """

    system_prompt = """
You extract structured information from employee time-off requests.

Extract these fields:

1. leave_type
2. start_date
3. end_date

Allowed leave types are:
- annual
- sick
- earned
- casual_sick

Return ONLY valid JSON in this exact format:

{
    "leave_type": "annual",
    "start_date": "2026-08-20",
    "end_date": "2026-08-22"
}

If a field is missing or cannot be determined, use null.

Do not invent dates or leave types.
"""

    response = client.chat.completions.create(
        model=deployment,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": message,
            },
        ],
    )

    content = response.choices[0].message.content

    return json.loads(content)


if __name__ == "__main__":

    from .. import config

    client = AzureOpenAI(
        azure_endpoint=config.AZURE_ENDPOINT,
        api_key=config.AZURE_API_KEY,
        api_version=config.AZURE_API_VERSION,
    )

    message = test_message = "I want to take some annual leave."

    result = extract_leave_request(
        client,
        config.AZURE_CHAT_DEPLOYMENT,
        message,
    )

    print(result)