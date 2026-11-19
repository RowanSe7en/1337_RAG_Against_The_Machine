from pydantic import BaseModel, Field
from typing import List

class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda:str(uuid.uuid4()))
    question: str

class AnsweredQuestion(UnansweredQuestion):
    sources: List
    answer: str

class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int