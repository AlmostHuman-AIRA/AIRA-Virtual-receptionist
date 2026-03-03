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
        Employee(name="Arjun", department="HR", cabin_number="201"),
        Employee(name="Meera", department="Finance", cabin_number="305"),
        Employee(name="Rohit", department="Engineering", cabin_number="110"),
        Employee(name="Kavya", department="Marketing", cabin_number="402"),
        Employee(name="Sanjay", department="Admin", cabin_number="101"),
        Employee(name="Neha", department="Engineering", cabin_number="112"),
        Employee(name="Vivek", department="Engineering", cabin_number="115"),
        Employee(name="Priya", department="HR", cabin_number="202"),
        Employee(name="Aman", department="Sales", cabin_number="501"),
        Employee(name="Ritu", department="Support", cabin_number="120"),
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
