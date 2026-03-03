from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    department = Column(String)
    cabin_number = Column(String)


class Meeting(Base):
    __tablename__ = "meetings"

    id = Column(Integer, primary_key=True)
    employee_name = Column(String)
    visitor_name = Column(String)
    meeting_time = Column(DateTime)


class Visitor(Base):
    __tablename__ = "visitors"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    meeting_with = Column(String)
    checkin_time = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="Checked-In")
