"""Date / time awareness and simple date arithmetic."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class DateTimeInput(BaseModel):
    offset_days: int = Field(
        0,
        description=(
            "Days from today. 0 = today, 7 = one week from now, -1 = yesterday. "
            "Use this to resolve phrases like 'in 10 days' or 'next Friday'."
        ),
    )


class DateTimeTool(BaseTool):
    name: str = "Date and Time"
    description: str = (
        "Returns the current date, time and weekday. Pass offset_days to get a "
        "date relative to today (e.g. offset_days=10 for a deadline in 10 days)."
    )
    args_schema: Type[BaseModel] = DateTimeInput

    def _run(self, offset_days: int = 0) -> str:
        now = datetime.now()
        target = date.today() + timedelta(days=offset_days)
        lines = [
            f"Now: {now.strftime('%A, %d %B %Y, %H:%M')}",
        ]
        if offset_days:
            direction = "from now" if offset_days > 0 else "ago"
            lines.append(
                f"{abs(offset_days)} day(s) {direction}: "
                f"{target.strftime('%A, %d %B %Y')}"
            )
        return "\n".join(lines)
