from datetime import datetime

from .database import SessionLocal
from .models import Employee, Meeting


def seed_database():
    db = SessionLocal()

    # Prevent reseeding
    if db.query(Employee).first():
        db.close()
        return

    employees = [
        Employee(name="Arjun", department="HR", cabin_number="201", role="HR Manager"),
        Employee(
            name="Meera",
            department="Finance",
            cabin_number="305",
            role="Financial Analyst",
        ),
        Employee(
            name="Rohit",
            department="Engineering",
            cabin_number="110",
            role="Software Engineer",
        ),
        Employee(
            name="Kavya",
            department="Marketing",
            cabin_number="402",
            role="Marketing Manager",
        ),
        Employee(
            name="Sanjay",
            department="Admin",
            cabin_number="101",
            role="Office Administrator",
        ),
        Employee(
            name="Neha",
            department="Engineering",
            cabin_number="112",
            role="Backend Engineer",
        ),
        Employee(
            name="Vivek",
            department="Engineering",
            cabin_number="115",
            role="DevOps Engineer",
        ),
        Employee(
            name="Priya", department="HR", cabin_number="202", role="HR Executive"
        ),
        Employee(
            name="Aman", department="Sales", cabin_number="501", role="Sales Manager"
        ),
        Employee(
            name="Ritu",
            department="Support",
            cabin_number="120",
            role="Customer Support Executive",
        ),
    ]

    meetings = [
        Meeting(
            employee_name="Arjun", visitor_name="Rahul", meeting_time=datetime.now()
        ),
        Meeting(
            employee_name="Meera", visitor_name="Anita", meeting_time=datetime.now()
        ),
        Meeting(
            employee_name="Rohit", visitor_name="Kiran", meeting_time=datetime.now()
        ),
    ]

    db.add_all(employees)
    db.add_all(meetings)
    db.commit()
    db.close()
