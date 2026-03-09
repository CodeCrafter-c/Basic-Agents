from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
import operator

model=ChatOllama(
    model="qwen2.5:7b"
)

class EvaluationShema(BaseModel):
    feedback:str=Field(description="Detailed feedback for essay.")
    score:int = Field(description="GIve score to essay from out of 10",ge=0,le=10)

structured_model=model.with_structured_output(EvaluationShema)

essay="""
Technology has become an essential part of modern education. It has transformed the way students learn and how teachers deliver knowledge. With the help of the internet, students can access a vast amount of information from anywhere in the world. Online resources such as educational videos, digital libraries, and interactive learning platforms make studying more engaging and effective.

Another major advantage of technology in education is improved communication. Students can easily interact with teachers through online platforms, discussion forums, and virtual classrooms. This allows learning to continue even outside the traditional classroom environment. Technology also enables personalized learning, where students can learn at their own pace using digital tools and adaptive learning systems.

However, technology also comes with certain challenges. Excessive use of digital devices can lead to distractions and reduced focus. Students may sometimes rely too heavily on online sources without developing critical thinking skills. Therefore, it is important for educators and students to maintain a balance between technology and traditional learning methods.

In conclusion, technology has greatly enhanced the education system by making learning more accessible, flexible, and interactive. When used wisely, it can significantly improve the quality of education and help students prepare for the future.
"""


class EssayState(TypedDict):
    topic:str
    essay:str
    language_feedback:str
    analysis_feedback:str
    clarity_feedback:str
    overall_feedback:str
    individual_scores:Annotated[list[int],operator.add]
    avg_score:float

def evaluate_language(state:EssayState)->EssayState:
    prompt=f"""
    Evaluate the language quality of the following essay  based on the topic of the essay and the essay itself and provide a clear feedback.
    And also give it a score out of 10.
    
    {state["topic"]}
    {state["essay"]}
    """
    output=structured_model.invoke(prompt)
    return {'language_feedback':output.feedback,'individual_scores':[output.score]}    
    
def evaluate_analysis(state:EssayState)->EssayState:
    prompt=f"""
    Evaluate the analysis  of the  depth of the following essay  based on the topic of the essay and the essay itself and provide a clear feedback.
    And also give it a score out of 10.
    
    {state["topic"]}
    {state["essay"]}
    """
    output=structured_model.invoke(prompt)
    return{"analysis_feedback":output.feedback,"individual_scores":[output.score]}

def evaluate_thought(state:EssayState)->EssayState:
    prompt=f"""
    Evaluate the clarity  of the  thought of the following essay  based on the topic of the essay and the essay itself and provide a clear feedback.
    And also give it a score out of 10.
    
    {state["topic"]}
    {state["essay"]}    
    """
    output=structured_model.invoke(prompt)
    return{"clarity_feedback":output.feedback,"individual_scores":[output.score]}

def final_evaluation(state:EssayState)->EssayState:
    prompt=f"""
    Based on the following feedback , create a summarised feedback.
    
    1. language feedback= {state["language_feedback"]} 
    2. Analysis feedback= {state["analysis_feedback"]} 
    3. Clarity_feedback = {state["clarity_feedback"]}
    
    """
    overall_feedback=model.invoke(prompt).content
    avg_score=sum(state["individual_scores"])/len(state["individual_scores"])    
    return{'overall_feedback':overall_feedback,"avg_score":avg_score}


graph=StateGraph(EssayState)

#nodes
graph.add_node("evaluate_language",evaluate_language)
graph.add_node("evaluate_analysis",evaluate_analysis)
graph.add_node("evaluate_thought",evaluate_thought)
graph.add_node("final_evaluation",final_evaluation)

#edges
graph.add_edge(START,"evaluate_language")
graph.add_edge(START,"evaluate_analysis")
graph.add_edge(START,"evaluate_thought")

graph.add_edge("evaluate_language","final_evaluation")
graph.add_edge("evaluate_analysis","final_evaluation")
graph.add_edge("evaluate_thought","final_evaluation")

graph.add_edge("final_evaluation",END)

#compile
workflow=graph.compile()

intialState={
    "topic":"The Importance of Technology in Modern Education",
    "essay":essay
}
print(workflow.invoke(intialState))






