import json

from .. import config


REQUESTS_FILE = config.DATA_DIR / "requests.json"


def get_requests(employee_id: str):
    """
    Return all time-off requests belonging to the selected employee.
    """

    requests = json.loads(
        REQUESTS_FILE.read_text()
    )

    employee_requests = [
        request
        for request in requests
        if request["employee_id"] == employee_id
    ]

    return {
        "success": True,
        "employee_id": employee_id,
        "requests": employee_requests,
    }


if __name__ == "__main__":

    print("Testing request tool...\n")

    result = get_requests("E001")

    print(result)



def submit_request(
    employee_id: str,
    leave_type: str,
    start_date: str,
    end_date: str,
):
    """
    Submit a new time-off request for an employee.
    """

    requests = json.loads(
        REQUESTS_FILE.read_text()
    )

    # Generate a new request ID
    if requests:
        numbers = [
            int(request["id"].split("-")[1])
            for request in requests
            if request["id"].startswith("REQ-")
        ]

        next_number = max(numbers) + 1 if numbers else 1001

    else:
        next_number = 1001

    request_id = f"REQ-{next_number}"

    new_request = {
        "id": request_id,
        "employee_id": employee_id,
        "leave_type": leave_type,
        "start_date": start_date,
        "end_date": end_date,
        "status": "pending",
    }

    requests.append(new_request)

    REQUESTS_FILE.write_text(
        json.dumps(requests, indent=2)
    )

    return {
        "success": True,
        "request": new_request,
    }


if __name__ == "__main__":

    print("Testing request tool...\n")

    result = get_requests("E001")
    print("Existing requests:")
    print(result)

    print("\nSubmitting test request...\n")

    result = submit_request(
        employee_id="E001",
        leave_type="annual",
        start_date="2026-08-25",
        end_date="2026-08-27",
    )

    print(result)

    