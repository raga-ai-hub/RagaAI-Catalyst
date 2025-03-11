import os
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import TypedDict, Annotated, List
import operator
import os
from dotenv import load_dotenv

load_dotenv()

from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer


catalyst = RagaAICatalyst(
    access_key=os.getenv['RAGAAICATALYST_ACCESS_KEY'], 
    secret_key=os.getenv('RAGAAICATALYST_SECRET_KEY'), 
    base_url=os.getenv('RAGAAICATALYST_BASE_URL')
)
# Initialize tracer
tracer = Tracer(
    project_name="example_testing",
    dataset_name="langgraph_research_assistant_trial_00",
    tracer_type="agentic/langchain",
)

init_tracing(catalyst=catalyst, tracer=tracer)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)  
tavily_tool = TavilySearchResults(max_results=5)  

# State structure
class ResearchState(TypedDict):
    topic: str  
    sub_questions: List[str]  
    answers: List[dict] 
    synthesis: str 
    criticism: str 
    iteration: Annotated[int, operator.add]  
    status: str

# Nodes
def generate_sub_questions(state: ResearchState) -> ResearchState:
    """Generate sub-questions based on the topic."""
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Given the topic '{topic}', generate 3 specific sub-questions to guide research."
    )
    response = llm.invoke(prompt.format(topic=state["topic"]))
    questions = [q.strip() for q in response.content.split("\n") if q.strip()]
    return {"sub_questions": questions, "status": "generated_questions"}

def research_sub_questions(state: ResearchState) -> ResearchState:
    """Research each sub-question using Tavily."""
    answers = []
    for question in state["sub_questions"]:
        search_results = tavily_tool.invoke(question)
        prompt = PromptTemplate(
            input_variables=["question", "search_results"],
            template="Answer '{question}' concisely based on: {search_results}"
        )
        answer = llm.invoke(prompt.format(
            question=question,
            search_results=[r["content"] for r in search_results]
        ))
        answers.append({
            "question": question,
            "answer": answer.content,
            "sources": [r["url"] for r in search_results]
        })
    return {"answers": answers, "status": "researched"}

def synthesize_findings(state: ResearchState) -> ResearchState:
    """Synthesize answers into a cohesive report."""
    prompt = PromptTemplate(
        input_variables=["topic", "answers"],
        template="Synthesize a 200-word report on '{topic}' using these findings:\n{answers}"
    )
    synthesis = llm.invoke(prompt.format(
        topic=state["topic"],
        answers="\n".join([f"Q: {a['question']}\nA: {a['answer']}" for a in state["answers"]])
    ))
    return {"synthesis": synthesis.content, "status": "synthesized"}

def critique_synthesis(state: ResearchState) -> ResearchState:
    """Critique the synthesis for completeness and accuracy."""
    prompt = PromptTemplate(
        input_variables=["topic", "synthesis", "answers"],
        template="Critique this report on '{topic}':\n{synthesis}\nBased on: {answers}\nReturn 'pass' or issues."
    )
    critique = llm.invoke(prompt.format(
        topic=state["topic"],
        synthesis=state["synthesis"],
        answers="\n".join([f"Q: {a['question']}\nA: {a['answer']}" for a in state["answers"]])
    ))
    return {"criticism": critique.content}

def refine_synthesis(state: ResearchState) -> ResearchState:
    """Refine the synthesis based on critique."""
    prompt = PromptTemplate(
        input_variables=["topic", "synthesis", "critique", "answers"],
        template="Refine this report on '{topic}':\n{synthesis}\nFix these issues: {critique}\nUsing: {answers}"
    )
    refined = llm.invoke(prompt.format(
        topic=state["topic"],
        synthesis=state["synthesis"],
        critique=state["critique"],
        answers="\n".join([f"Q: {a['question']}\nA: {a['answer']}" for a in state["answers"]])
    ))
    return {"synthesis": refined.content, "iteration": state["iteration"] + 1, "status": "refined"}

# Conditional logic
def should_refine(state: ResearchState) -> str:
    if "pass" in state["criticism"].lower() or state["iteration"] >= 2:
        return "end"
    return "refine"

# State graph
workflow = StateGraph(ResearchState)
workflow.add_node("generate", generate_sub_questions)
workflow.add_node("research", research_sub_questions)
workflow.add_node("synthesize", synthesize_findings)
workflow.add_node("critique", critique_synthesis)
workflow.add_node("refine", refine_synthesis)

# Workflow
workflow.set_entry_point("generate")
workflow.add_edge("generate", "research")
workflow.add_edge("research", "synthesize")
workflow.add_edge("synthesize", "critique")
workflow.add_conditional_edges(
    "critique",
    should_refine,
    {"refine": "refine", "end": END}
)
workflow.add_edge("refine", "critique")

# Compile and run
app = workflow.compile()
initial_state = {
    "topic": "Impact of AI on healthcare by 2030",
    "iteration": 0,
    "status": "start"
}
print("Starting the Personal Research Assistant...")
with tracer:
    result = app.invoke(initial_state)

# Print results
print("\nFinal Research Report:")
print(f"Topic: {result['topic']}")
print("Sub-Questions and Answers:")
for ans in result["answers"]:
    print(f"- Q: {ans['question']}")
    print(f"  A: {ans['answer']}")
    print(f"  Sources: {ans['sources']}")
print(f"\nSynthesis:\n{result['synthesis']}")
print(f"\nCritique: {result['criticism']}")
print(f"Iterations: {result['iteration']}")