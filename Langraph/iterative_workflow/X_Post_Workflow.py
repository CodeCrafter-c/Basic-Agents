from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Literal,Annotated
import operator
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

genrator_llm=ChatOllama(model="qwen2.5:7b")
evaluator_llm=ChatOllama(model="qwen2.5:7b")
optimizor_llm=ChatOllama(model="qwen2.5:7b")

class TweetEvaluation(BaseModel):
    evaluation:Literal["approved", "needs_improvement"] = Field(..., description="Final evaluation result.")
    feedback: str = Field(..., description="feedback for the tweet.")

structured_model=evaluator_llm.with_structured_output(TweetEvaluation)

#state
class TweetState(TypedDict):
    topic:str
    tweet:str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iteration: int
    max_iteration: int
    
    tweet_history: Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]

def generate_tweet(state:TweetState)->TweetState:
    #prompt
    messages = [
        SystemMessage(content="You are a funny and clever Twitter/X influencer."),
        HumanMessage(content=f"""
Write a short, original, and hilarious tweet on the topic: "{state['topic']}".

Rules:
- Do NOT use question-answer format.
- Max 280 characters.
- Use observational humor, irony, sarcasm, or cultural references.
- Think in meme logic, punchlines, or relatable takes.
- Use simple, day to day english
""")
    ]
    
    res=genrator_llm.invoke(messages).content
    return {
    "tweet": res,
    "tweet_history": [res]
}

def Evaluate_tweet(state:TweetState)->TweetState:
    messages = [
    SystemMessage(content="You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format."),
    HumanMessage(content=f"""
Evaluate the following tweet:

Tweet: "{state['tweet']}"

Use the criteria below to evaluate the tweet:

1. Originality – Is this fresh, or have you seen it a hundred times before?  
2. Humor – Did it genuinely make you smile, laugh, or chuckle?  
3. Punchiness – Is it short, sharp, and scroll-stopping?  
4. Virality Potential – Would people retweet or share it?  
5. Format – Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?

Auto-reject if:
- It's written in question-answer format (e.g., "Why did..." or "What happens when...")
- It exceeds 280 characters
- It reads like a traditional setup-punchline joke
- Dont end with generic, throwaway, or deflating lines that weaken the humor (e.g., “Masterpieces of the auntie-uncle universe” or vague summaries)

### Respond ONLY in structured format:
- evaluation: "approved" or "needs_improvement"  
- feedback: One paragraph explaining the strengths and weaknesses 
""")
]

    res=structured_model.invoke(messages)
    return{
        "evaluation":res.evaluation,
        "feedback":res.feedback,
        "feedback_history":[res.feedback]
    }

def optimize_tweet(state: TweetState)->TweetState:
    messages = [
        SystemMessage(content="You punch up tweets for virality and humor based on given feedback."),
        HumanMessage(content=f"""
Improve the tweet based on this feedback:
"{state['feedback']}"

Topic: "{state['topic']}"
Original Tweet:
{state['tweet']}

Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
""")
    ]
    
    res=optimizor_llm.invoke(messages).content
    iteration=state['iteration']+1
    return {
        "tweet":res,
        "iteration":iteration,
        "tweet_history":[res]
    }


def route_evaluation(state: TweetState)->Literal["approved","needs_improvement"]:
    if(state["evaluation"]=='approved'or state["iteration"]>=state["max_iteration"]):
        return "approved"
    else:
        return "needs_improvement"


graph=StateGraph(TweetState)

#nodes
graph.add_node("generate",generate_tweet)
graph.add_node("evaluate",Evaluate_tweet)
graph.add_node("optimize",optimize_tweet)

#edges
graph.add_edge(START,"generate")
graph.add_edge("generate","evaluate")
graph.add_conditional_edges("evaluate",route_evaluation,{"approved" :END,"needs_improvement":"optimize"})
graph.add_edge("optimize","evaluate")

workflow=graph.compile()

initial_state = {
    "topic": "indian railway",
    "iteration": 1,
    "max_iteration": 5,
    "tweet_history": [],
    "feedback_history": []
}
final_state=workflow.invoke(initial_state)
print(final_state)


