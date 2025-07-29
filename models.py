from typing import Optional
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String
from database import Base


class MsgPayload(BaseModel):
    msg_id: Optional[int]
    msg_name: str


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    password = Column(String(255), nullable=False)
    resume_path = Column(String(255), nullable=True)  # New field
