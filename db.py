# db.py
from datetime import datetime
from sqlalchemy import ForeignKey, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship
import os

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    api_key: Mapped[str] = mapped_column(unique=True)
    email: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    requests: Mapped[list["APIRequest"]] = relationship(back_populates="user")

class APIRequest(Base):
    __tablename__ = "api_requests"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    endpoint: Mapped[str]
    timestamp: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    response_time_ms: Mapped[float]
    status_code: Mapped[int]

    user: Mapped[User] = relationship(back_populates="requests")

engine = create_engine(os.environ.get("DATABASE_URL", "sqlite:///ai_api.db"))
Base.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
