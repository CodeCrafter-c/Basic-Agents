from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Literal
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

model= ChatOllama(
    model="qwen2.5:7b"
)

class sentimentSchema(BaseModel):
    sentiment:Literal["positive","negative"]=Field(description="Sentiment of the review")
    
structured_model=model.with_structured_output(sentimentSchema)

class DiagnosisSchema(BaseModel):
    issue_type:Literal["UX","Performace","Bug","Support","Other"]=Field(description="the category of issue mentioned in the review")
    tone:Literal["angry","frustrated","calm","disappointed"]=Field(description="The emotional tone expressed by the user")
    urgency:Literal["low","medium","high"]=Field(description="How urgent or critical the issue appear to be")

diagnostic_model=model.with_structured_output(DiagnosisSchema)

class ReviewState(TypedDict):
    review:str
    sentiment:Literal["positive","negative"]
    diagnosis:dict
    response:str


    
def find_sentiment(state: ReviewState) -> ReviewState:
    prompt = f"""
    Determine whether the following review is positive or negative.

    Review:
    {state["review"]}
    """

    res = structured_model.invoke(prompt)

    return {"sentiment": res.sentiment}

def check_condition(state: ReviewState) -> Literal["run_diagnosis","positive_response"]:
    if state["sentiment"] == "negative":
        return "run_diagnosis"
    else:
        return "positive_response"

def positive_response(state:ReviewState)->ReviewState:
    prompt=f"""
    Write a warm thank you message in response to this review:

    review: {state["review"]}.
    
    Also ask the user to leave the feedback on our website.
    """
    
    response=model.invoke(prompt).content
    return {"response":response}

def negative_response(state:ReviewState)->ReviewState:
    prompt=f"""
    You are a support assistant
    
    The user had a {state["diagnosis"]["issue_type"]} issue, sounded like {state["diagnosis"]["tone"]}, and marked urgency is {state["diagnosis"]["urgency"]}.
    
    Write an empathetic , helpful and resolution message.
    """
    res=model.invoke(prompt).content
    return {"response":res}

def run_diagnosis(state:ReviewState)->ReviewState:
    prompt=f"""
    Diagnose the following  negaitve review.
    provide: 1. issue-type
             2. tone
             3. urgency

    review : {state["review"]}
    """
    res=diagnostic_model.invoke(prompt)
    
    return {"diagnosis":res.model_dump()}




#graph
graph=StateGraph(ReviewState)

#nodes
graph.add_node("find_sentiment",find_sentiment)
graph.add_node("run_diagnosis",run_diagnosis)
graph.add_node("positive_response",positive_response)
graph.add_node("negative_response",negative_response)


#edges
graph.add_edge(START,"find_sentiment")
graph.add_conditional_edges("find_sentiment",check_condition)

graph.add_edge("run_diagnosis","negative_response")
graph.add_edge("negative_response",END)
graph.add_edge("positive_response",END)


workflow=graph.compile()

intial_state={
    "review":""
}
final_state=workflow.invoke(intial_state)
print(final_state)