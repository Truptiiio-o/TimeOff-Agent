import json
from datetime import date

from .. import config
from .balance_tool import get_balance


REQUESTS_FILE = config.DATA_DIR / "requests.json"


def submit_request(
    employee_id: str,
    leave_type: str,
    start_date: str,
    end_date: str,
):
    """
    Create a new time-off request for the selected employee.

    The request is saved only if:
    - the dates are valid
    - the end date is not before the start date
    - the employee has enough leave balance
    """

    # ---------------------------------------------------------
    # 1. Validate the dates
    # ---------------------------------------------------------

    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)

    except ValueError:
        return {
            "success": False,
            "message": "Invalid date format. Use YYYY-MM-DD.",
        }

    if end < start:
        return {
            "success": False,
            "message": "End date cannot be before start date.",
        }

    # ---------------------------------------------------------
    # 2. Calculate requested number of days
    # ---------------------------------------------------------

    requested_days = (end - start).days + 1

    # ---------------------------------------------------------
    # 3. Check employee's leave balance
    # ---------------------------------------------------------

    balance_result = get_balance(
        employee_id,
        leave_type,
    )

    if not balance_result["success"]:
        return {
            "success": False,
            "message": (
                f"No balance found for {leave_type} leave "
                "for this employee."
            ),
        }

    remaining_days = balance_result["remaining_days"]

    if requested_days > remaining_days:
        return {
            "success": False,
            "message": (
                f"Insufficient {leave_type} leave balance. "
                f"Requested {requested_days} days, "
                f"but only {remaining_days} days remain."
            ),
        }

    # ---------------------------------------------------------
    # 4. Load existing requests
    # ---------------------------------------------------------

    requests = json.loads(
        REQUESTS_FILE.read_text()
    )

    # ---------------------------------------------------------
    # 5. Generate a new request ID
    # ---------------------------------------------------------

    request_number = 1000 + len(requests) + 1

    request_id = f"REQ-{request_number}"

    # ---------------------------------------------------------
    # 6. Create the request
    # ---------------------------------------------------------

    new_request = {
        "id": request_id,
        "employee_id": employee_id,
        "leave_type": leave_type,
        "start_date": start_date,
        "end_date": end_date,
        "status": "pending",
    }

    # ---------------------------------------------------------
    # 7. Save the request
    # ---------------------------------------------------------

    requests.append(new_request)

    REQUESTS_FILE.write_text(
        json.dumps(
            requests,
            indent=2,
        )
    )

    # ---------------------------------------------------------
    # 8. Return success only after saving
    # ---------------------------------------------------------

    return {
        "success": True,
        "request": new_request,
    }


if __name__ == "__main__":

    result = submit_request(
        employee_id="E001",
        leave_type="annual",
        start_date="2026-08-01",
        end_date="2026-08-20",
    )

    print(result)