import json
from pathlib import Path

from .. import config


BALANCES_FILE = config.DATA_DIR / "balances.json"


def get_balance(employee_id: str, leave_type: str | None = None):
    """
    Get the leave balance for the currently selected employee.

    employee_id:
        The employee whose balance is being requested.

    leave_type:
        Optional type of leave, such as annual or sick.

        If omitted, all available leave balances for the employee
        are returned.
    """

    balances = json.loads(
        BALANCES_FILE.read_text()
    )

    employee_balances = [
        balance
        for balance in balances
        if balance["employee_id"] == employee_id
    ]

    if not employee_balances:
        return {
            "success": False,
            "message": "No balance information found for this employee."
        }

    # If a specific leave type was requested
    if leave_type:

        matching_balance = [
            balance
            for balance in employee_balances
            if balance["leave_type"].lower() == leave_type.lower()
        ]

        if not matching_balance:
            return {
                "success": False,
                "message": (
                    f"No {leave_type} leave balance was found "
                    "for this employee."
                )
            }

        balance = matching_balance[0]

        return {
            "success": True,
            "employee_id": employee_id,
            "leave_type": balance["leave_type"],
            "remaining_days": balance["remaining_days"],
        }

    # If no specific leave type was requested,
    # return all balances.
    return {
        "success": True,
        "employee_id": employee_id,
        "balances": employee_balances,
    }


if __name__ == "__main__":

    print("Testing balance tool...\n")

    result = get_balance("E002", "earned")

    print(result)