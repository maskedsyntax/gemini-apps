# news_generator_crew.py
import os

from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI

from crewai import Agent, Crew, Process, Task

# ------------------------------------------------------------------
# Configuration (set your API keys in environment variables or here)
# ------------------------------------------------------------------
# Obtain a free Serper API key at https://serper.dev
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

# OpenAI is used as the default LLM (you can replace with Groq, Anthropic, Ollama, etc.)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Optional: use a different model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ------------------------------------------------------------------
# Tools
# ------------------------------------------------------------------
search_tool = SerperDevTool(
    n_results=10,  # Number of search results to retrieve
    search_type="news",  # Specialised news search (uses https://google.serper.dev/news)
    country="us",  # Optional: restrict to a country
    locale="en",  # Optional: language/locale
)

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Agents
# ------------------------------------------------------------------
researcher = Agent(
    role="Senior News Researcher",
    goal="Find the most recent, credible, and important news stories about {topic} from the last 7 days",
    backstory=(
        "You are an expert investigative journalist with decades of experience. "
        "You excel at locating timely, factual sources and extracting key details."
    ),
    tools=[search_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

writer = Agent(
    role="Professional News Editor & Writer",
    goal="Write a clear, neutral, engaging news article based on the latest research",
    backstory=(
        "You are a seasoned editor for a major international news outlet. "
        "You write concise, objective articles with proper structure (title, lead, body, sources)."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# ------------------------------------------------------------------
# Tasks
# ------------------------------------------------------------------
research_task = Task(
    description=(
        "Search for the very latest news (last 7 days) on the topic: {topic}. "
        "Focus only on reputable sources. "
        "Summarize the top 5-8 most important stories with title, publication date, source, link, and a brief key takeaway. "
        "Include a short overall summary of the current situation."
    ),
    expected_output="A bullet-point list of the top recent news items with summaries and an overall context paragraph.",
    agent=researcher,
    output_file="research_output.md",  # Optional: saves intermediate research
)

write_task = Task(
    description=(
        "Using the research provided, write a professional news article (800-1200 words) about {topic}. "
        "Structure: catchy title, strong lead paragraph, detailed body with facts, neutral tone, and a short conclusion/outlook. "
        "Cite sources inline with links. End with a 'Sources' section. "
        "Write in markdown format."
    ),
    expected_output="A complete, publication-ready news article in markdown.",
    agent=writer,
    output_file="final_news_article.md",
)

# ------------------------------------------------------------------
# Crew (orchestration)
# ------------------------------------------------------------------
news_crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,  # Researcher → Writer
    verbose=True,
    memory=True,  # Remembers context between tasks
    cache=True,
)

# ------------------------------------------------------------------
# Run the crew
# ------------------------------------------------------------------
if __name__ == "__main__":
    topic = (
        input(
            "Enter the news topic (e.g., 'US election 2024 results', 'AI regulation Europe'): "
        )
        or "latest breakthroughs in quantum computing"
    )

    result = news_crew.kickoff(inputs={"topic": topic})

    print("\n" + "=" * 60)
    print("NEWS ARTICLE GENERATED AND SAVED TO 'final_news_article.md'")
    print("=" * 60)
    print(result)
