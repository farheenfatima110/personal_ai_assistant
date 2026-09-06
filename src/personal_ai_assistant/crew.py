from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from .tools.calculator_tool import CalculatorTool
from .tools.datetime_tool import DateTimeTool
from .tools.document_tool import DocumentTool
from .tools.personal_memory_tool import PersonalMemoryTool, SaveMemoryTool
from .tools.study_material_tool import StudyMaterialTool
from .tools.weather_tool import WeatherTool
from .tools.web_search_tool import WebSearchTool
from .tools.wikipedia_tool import WikipediaTool


def _has_answer(output) -> tuple[bool, str]:
    """Guardrail: reject empty / pure-refusal answers so the task retries."""
    text = (getattr(output, "raw", None) or str(output)).strip()
    if len(text) < 2:
        return (False, "The answer is empty. Provide a direct response to the request.")
    lowered = text.lower()
    if lowered.startswith(("i cannot", "i can't", "as an ai")) and len(text) < 60:
        return (False, "Use the available tools to give a useful answer instead of refusing.")
    return (True, text)


@CrewBase
class PersonalAiAssistant:
    """Student Study & Productivity Assistant.

    A single ReAct agent equipped with eight tools (RAG over the student's notes,
    web search, Wikipedia, weather, date math, safe calculator, and read/write
    personal memory). It reads the request, picks the right tool(s), and answers.
    """

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def assistant(self) -> Agent:
        return Agent(
            config=self.agents_config["assistant"],  # type: ignore[index]
            tools=[
                StudyMaterialTool(),
                DocumentTool(),
                WebSearchTool(),
                WikipediaTool(),
                WeatherTool(),
                DateTimeTool(),
                CalculatorTool(),
                PersonalMemoryTool(),
                SaveMemoryTool(),
            ],
            inject_date=True,
            verbose=True,
            max_iter=12,
        )

    @task
    def assist_task(self) -> Task:
        return Task(
            config=self.tasks_config["assist_task"],  # type: ignore[index]
            guardrail=_has_answer,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Personal AI Assistant crew."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            output_log_file="logs.txt",
        )
